import timm
import torch.nn as nn
from .labels import SINGLE_HEADS, MULTI_KEYS


class TongueNet(nn.Module):
    """共享骨干 + 多任务头。"""
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.heads = nn.ModuleDict({h: nn.Linear(d, len(c)) for h, c in SINGLE_HEADS.items()})
        self.heads["multi"] = nn.Linear(d, len(MULTI_KEYS))

    def forward(self, x):
        f = self.backbone(x)
        return {k: m(f) for k, m in self.heads.items()}
