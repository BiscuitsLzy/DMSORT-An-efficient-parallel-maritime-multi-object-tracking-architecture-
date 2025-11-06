import timm
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Dict
import math

class LiTAE(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.backbone = timm.create_model(
            cfg["model"]["type"],
            pretrained=False,
            num_classes=0
        )

        # 加载预训练权重
        if cfg["model"]["pretrained"] and cfg["model"]["path"]:
            state_dict = torch.load(cfg["model"]["path"], map_location='cpu',weights_only=True)
            #self.backbone.load_state_dict(state_dict, strict=False)

            # 统计匹配的键和未匹配的键
            model_params = self.backbone.state_dict()
            matched_params = []
            unmatched_params = []

            for key in state_dict:
                if key in model_params:
                    matched_params.append(key)
                else:
                    unmatched_params.append(key)

            print(f"Matched parameters: {len(matched_params)}")
            print(f"Unmatched parameters: {len(unmatched_params)}")
            print("unMatched parameters:", unmatched_params)

            self.backbone.load_state_dict(state_dict, strict=False)

        self.projector = nn.Linear(
            self.backbone.num_features,
            cfg["model"]["feature_dim"]
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.projector(features)
        return F.normalize(features, p=2, dim=1)

def build_model(cfg: Dict) -> nn.Module:
    model_dict = {
        "litae" :LiTAE,
    }
    return model_dict[cfg['model']['name']](cfg)


