import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms
import numpy as np

class ClipB16FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipB16FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x):
        # x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
    
    def vforward(self, x, enhance=True, both=False):
        inputs = dict(pixel_values=self.normalizer(x))
        pixel_values = inputs["pixel_values"].unsqueeze(0) if inputs["pixel_values"].ndim == 3 else inputs["pixel_values"]
        pixel_values = pixel_values.to(next(self.model.parameters()).device)

        vision_model = self.model.vision_model

        outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs[2]  # tuple of hidden states

        layer_input = hidden_states[-2]  # [1, tokens, hidden_dim]
        last_layer = vision_model.encoder.layers[-1]

        norm_x = last_layer.layer_norm1(layer_input)
        q = last_layer.self_attn.q_proj(norm_x)
        k = last_layer.self_attn.k_proj(norm_x)
        v = last_layer.self_attn.v_proj(norm_x)

        num_heads = last_layer.self_attn.num_heads
        head_dim = last_layer.self_attn.head_dim
        q = q.view(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)  # [1, heads, tokens, head_dim]
        k = k.view(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)

        if enhance:
            attn_scores = (v @ v.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = attn_weights @ v  # [1, heads, tokens, head_dim]
        else:
            attn_output = v

        attn_output_concat = attn_output.permute(0, 2, 1, 3).reshape(1, -1, num_heads * head_dim)
        proj_v = last_layer.self_attn.out_proj(attn_output_concat)  # [1, tokens, hidden_dim]

        post_ln = vision_model.post_layernorm
        vis_proj = self.model.visual_projection

        image_embeds = post_ln(proj_v)              # [1, tokens, hidden_dim]
        image_embeds = vis_proj(image_embeds)       # [1, tokens, text_dim]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        if both:
            x_feat = post_ln(hidden_states[-1])
            x_feat = vis_proj(x_feat)
            x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
            return image_embeds, x_feat
        else:
            return image_embeds

    def xforward(self, x, enhance=True):
        inputs = dict(pixel_values=self.normalizer(x))
        pixel_values = inputs["pixel_values"].unsqueeze(0) if inputs["pixel_values"].ndim == 3 else inputs["pixel_values"]
        pixel_values = pixel_values.to(next(self.model.parameters()).device)

        vision_model = self.model.vision_model

        outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs[2]  # tuple of hidden states

        x_last = hidden_states[-1]  # [1, tokens, hidden_dim]

        post_ln = vision_model.post_layernorm
        vis_proj = self.model.visual_projection

        image_embeds = post_ln(x_last)              # [1, tokens, hidden_dim]
        image_embeds = vis_proj(image_embeds)       # [1, tokens, text_dim]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds
    
    def tforward(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs.to(self.model.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features
