from pydoc import text
import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable, Dict
import torch.nn.functional as F

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {}  
        for i, model in enumerate(self.extractors):
            features[i] = model(x)
        return features
    
    def vforward(self, x: Tensor, enhance, both=False) -> Tensor:
        features = {}  
        if not both:
            for i, model in enumerate(self.extractors):
                features[i] = model.vforward(x, enhance)
            return features
        else:
            x_feat = {}
            for i, model in enumerate(self.extractors):
                features[i], x_feat[i] = model.vforward(x, enhance, both)
            return features, x_feat
    
    def tforward(self, text: list) -> Tensor:
        features = {} 
        for i, model in enumerate(self.extractors):
            features[i] = model.tforward(text)
        return features
    
    def xforward(self, x: Tensor) -> Tensor:
        features = {}  
        for i, model in enumerate(self.extractors):
            features[i] = model.xforward(x)
        return features

class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        # source features
        self.source_text = []
        self.source_image = []
        self.source_value = []      
        self.source_features = []   
        # target features
        self.target_text = []
        self.mask = [] # nums of model * [B, N-1]
        self.mask_index = []
        
        self.enhance = False

    @torch.no_grad()
    def set_enhance(self, x: bool):
        self.enhance = x


    #untargeted attack
    @torch.no_grad()
    def set_ground_truth(self, x: Tensor, text, vattack):
        self.source_image.clear()
        self.source_text.clear()
        self.source_value.clear()
        self.source_features.clear()

        for model in self.extractors:

            self.source_value.append(model.vforward(x, enhance=self.enhance))
            self.source_features.append(model.xforward(x))

            if vattack:
                self.source_image.append(model.vforward(x, enhance=self.enhance))
            else:
                self.source_image.append(model.xforward(x))
        
        
        self.source_text = text

    # targeted attack
    @torch.no_grad()
    def set_target_text(self, text):
        self.target_text.clear()
        self.target_text = text

    @torch.no_grad()
    def set_mask(self):
        self.mask.clear()
        
        # image B N C ,text B D
        if self.source_image and self.source_text:
            for index, model in enumerate(self.extractors):
                B, N, C = self.source_image[index].shape

                patch_tokens = self.source_image[index][:, 1:, :]
                text_features = self.source_text[index].unsqueeze(1).expand(-1, N-1, -1)
                
                similarities = F.cosine_similarity(patch_tokens, text_features, dim=2)
                max_sim = similarities.max(dim=1, keepdim=True)[0]
                min_sim = similarities.min(dim=1, keepdim=True)[0]
                threshold = (max_sim + min_sim) / 2

                mask = similarities > threshold
                self.mask.append(mask)
        else:
            raise ValueError("Please set the source image and source text first.")
 
    @torch.no_grad()
    def set_mask_index(self):
        self.mask_index.clear()
        # image B N C ,text B D
        if self.source_features and self.source_text:
            for index, model in enumerate(self.extractors):
                B, N, C = self.source_features[index].shape

                patch_tokens = self.source_features[index][:, 1:, :]
                # text_features = self.source_text[index].unsqueeze(1).expand(-1, N-1, -1)
                text_features = self.source_features[index][:, 0, :]
                
                similarities = F.cosine_similarity(patch_tokens, text_features, dim=2)
                max_sim = similarities.max(dim=1, keepdim=True)[0]
                min_sim = similarities.min(dim=1, keepdim=True)[0]
                # threshold = torch.mean(similarities)
                threshold = ((max_sim - min_sim) / 4 * 3) + min_sim

                mask = similarities > threshold
                self.mask_index.append(mask)
                # print(torch.sum(mask))  
        else:
            raise ValueError("Please set the source image and source text first.")


    def __call__(self, feature_dict: Dict[int, Tensor], x_feature_dict: Dict[int, Tensor] = None, Vision_A: Any = None, Target_A: Any = None) -> Tensor:

        loss = 0

        for index, model in enumerate(self.extractors):
            # print(feature_dict[index].shape)
            feature = feature_dict[index][:,1:,:] # B N C 去掉CLS Token      
            gt_text = self.source_text[index].unsqueeze(1).expand(-1, feature.shape[1], -1) # B N D
            txt_loss = - torch.mean(F.cosine_similarity(feature, gt_text, dim=2)*self.mask[index]) # 
            loss +=  txt_loss

            if x_feature_dict:
                x_feat = x_feature_dict[index][:,1:,:] # B N C 
                x_txt_loss = - torch.mean(F.cosine_similarity(x_feat, gt_text, dim=2)*self.mask_index[index])
                loss += x_txt_loss
        
            if Target_A: # targeted attack

                gt_text = self.target_text[index].unsqueeze(1).expand(-1, feature.shape[1], -1) # B N D
                if self.target_text:
                    txt_loss = torch.mean(F.cosine_similarity(feature, gt_text, dim=2)*self.mask[index])
                    loss += txt_loss
                
                if x_feature_dict:
                    x_feat = x_feature_dict[index][:,1:,:] # B N C 
                    x_txt_loss = torch.mean(F.cosine_similarity(x_feat, gt_text, dim=2)*self.mask_index[index])
                    loss += x_txt_loss

        loss = loss / len(self.extractors)

        return loss
