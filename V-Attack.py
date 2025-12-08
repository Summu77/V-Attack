from calendar import c
import os
import json
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image
import hydra
from omegaconf import DictConfig
import os
from config.config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional
from torch import nn, no_grad
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)

from utils import ensure_dir

import pandas as pd

def load_labels_from_csv(csv_path, label_column="label"):
        
    df = pd.read_csv(csv_path)

    if label_column not in df.columns:
        raise ValueError(f"文件中未找到 {label_column} 列，实际列名: {list(df.columns)}")

    labels = df[label_column].dropna().astype(str).tolist()

    return labels

# Mapping from backbone names to model classes
BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}

def get_models(cfg: MainConfig):
    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)  
    ensemble_extractor = EnsembleFeatureExtractor(models)
    return ensemble_extractor, models


def get_ensemble_loss(models: List[nn.Module]):
    ensemble_loss = EnsembleFeatureLoss(models)
    return ensemble_loss

def set_environment(seed=2026):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Transform PIL.Image to PyTorch Tensor
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

# Dataset with image paths
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)

def attack_imgpair(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    image_org: torch.Tensor,
    path_org: List[str],
    source_text,
    target_text,
):
    image_org= image_org.to(cfg.model.device)
    attack_type = cfg.attack.method
    attack_fn = {"pgd": pgd_attack,}[attack_type]

    adv_image = attack_fn(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        image_org=image_org,
        source_text=source_text,
        target_text=target_text
    )

    for path_idx in range(len(path_org)):

        folder = os.path.basename(os.path.dirname(os.path.dirname(path_org[path_idx])))
        name = os.path.basename(path_org[path_idx])
        st_index = cfg.data.num_samples_index
        ed_index = cfg.data.num_samples_index + cfg.data.num_samples
        
        if cfg.attack.target_text:
            folder += "_targeted"
        else:
            folder += "_untargeted"

        if cfg.attack.vattack:
            folder += "_v"
        else:
            folder += "_x"

        if cfg.attack.vision_attack:
            folder_to_save = os.path.join(cfg.data.output, folder, f"{st_index}_{ed_index}_with_vision")
        else:
            if not cfg.attack.enhance:
                folder_to_save = os.path.join(cfg.data.output, folder, f"{st_index}_{ed_index}_no_vision_no_enhance")
            else:
                folder_to_save = os.path.join(cfg.data.output, folder, f"{st_index}_{ed_index}_no_vision")

        ensure_dir(folder_to_save)

        ext = os.path.splitext(name)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            new_name = os.path.splitext(name)[0] + ".png"
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, new_name)
            )
        elif ext == ".png":
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name)
            )
        else:
            print(f"Warning: Unsupported file format {ext} in {name}")

def dict_to_list(features_dict):
    length = len(features_dict)
    feature_list = []
    for idx in range(length):
        feature_list.append(features_dict[idx])
    return feature_list

def show_mask_grid(mask, name = "1", ncols=4, cmap="viridis"):

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    B, D = mask.shape

    mask_imgs = mask.reshape(B, int(D**0.5), int(D**0.5))

    nrows = int(np.ceil(B / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    for i in range(B):
        ax = axes[i]
        im = ax.imshow(mask_imgs[i], cmap=cmap)
        ax.set_title(f"Mask {i}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(B, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(f"output_{name}.png")

def pgd_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    image_org: torch.Tensor,
    source_text,
    target_text,
):

    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    with torch.no_grad():
        ensemble_loss.set_enhance(cfg.attack.enhance)
        ensemble_loss.set_ground_truth(source_crop(image_org).to(cfg.model.device), source_text, cfg.attack.vattack)
        ensemble_loss.set_target_text(target_text)
        ensemble_loss.set_mask()
        ensemble_loss.set_mask_index()

    # Main optimization loop
    for epoch in pbar:         
        # Forward pass
        adv_image = image_org + delta

        # If using source crop, calculate additional local similarity
        local_cropped = source_crop(adv_image)
        if cfg.attack.vattack:
            if not cfg.attack.both:
                local_features = ensemble_extractor.vforward(local_cropped, enhance=cfg.attack.enhance)
                local_features = dict_to_list(local_features)
                loss = - ensemble_loss(local_features, Vision_A=cfg.attack.vision_attack, Target_A=cfg.attack.target_text)
            else:
                local_features, x_features = ensemble_extractor.vforward(local_cropped, enhance=cfg.attack.enhance, both=cfg.attack.both)
                local_features = dict_to_list(local_features)
                loss = - ensemble_loss(local_features, x_features, cfg.attack.vision_attack, cfg.attack.target_text)
        else:
            local_features = ensemble_extractor.xforward(local_cropped)
            loss = - ensemble_loss(local_features, Vision_A=cfg.attack.vision_attack, Target_A=cfg.attack.target_text)
            
        optimizer.zero_grad()
        loss.backward()

        # PGD update
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image

@hydra.main(version_base=None, config_path="config", config_name="ensemble")
def main(cfg: MainConfig):

    set_environment()

    ensemble_extractor, models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(models)

    ## load image info
    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)

    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop
        else torch.nn.Identity()
    )

    ## load text info
    source_text = []
    target_text = []

    if cfg.attack.source_text:
        source_text = load_labels_from_csv(cfg.data.text_path, label_column="source")

    if cfg.attack.target_text:   
        target_text = load_labels_from_csv(cfg.data.text_path, label_column="target") 
    
    for i, (image_org, _, path_org) in enumerate(data_loader_imagenet):

        if i < cfg.data.num_samples_index:
            continue

        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples_index + cfg.data.num_samples:
            break

        print(f"\nProcessing image {i+1}/{cfg.data.num_samples//cfg.data.batch_size}")

        if cfg.attack.source_text: 
            a = [source_text[i]]
            a = dict_to_list(ensemble_extractor.tforward(a)) # {[1, 768]} -> list
        else:
            a = []
            
        if cfg.attack.target_text: 
            b = [target_text[i]]
            b = dict_to_list(ensemble_extractor.tforward(b)) # {[1, 768]} -> list
        else:
            b = [] 

        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            image_org=image_org,
            path_org=path_org,
            source_text = a,
            target_text = b,
        )

if __name__ == "__main__":
    main()