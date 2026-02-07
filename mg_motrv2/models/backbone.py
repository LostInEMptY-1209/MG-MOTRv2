"""
Backbone networks for MG-MOTRv2
多粒度骨干网络 - 提取多尺度特征

架构：ResNet + FPN → 多粒度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4
    
    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch * 4)
        
        self.downsample = None
        if stride != 1 or in_ch != mid_ch * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch * 4, 1, stride, bias=False),
                nn.BatchNorm2d(mid_ch * 4)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """
    ResNet骨干网络
    输出多尺度特征用于后续多粒度处理
    """
    def __init__(
        self,
        layers: List[int] = [3, 4, 6, 3],  # ResNet50
        in_channels: int = 3,
        base_channels: int = 64
    ):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, 7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet阶段
        self.in_channels = base_channels
        self.stages = nn.ModuleList()
        strides = [1, 2, 2, 2]
        channels = [base_channels * 4, base_channels * 8, base_channels * 16, base_channels * 32]
        
        for num_blocks, stride, out_ch in zip(layers, strides, channels):
            stage = self._make_stage(out_ch, num_blocks, stride)
            self.stages.append(stage)
    
    def _make_stage(self, out_ch: int, num_blocks: int, stride: int) -> nn.Module:
        """构建ResNet阶段"""
        blocks = [Bottleneck(self.in_channels, out_ch // 4, stride)]
        self.in_channels = out_ch
        for _ in range(1, num_blocks):
            blocks.append(Bottleneck(self.in_channels, out_ch // 4))
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            features: 多尺度特征字典，stride_8/16/32
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # stride 4
        
        features = {}
        stride = 4
        
        for stage in self.stages:
            x = stage(x)
            stride *= 2
            if stride in [8, 16, 32]:
                features[f"stride_{stride}"] = x
        
        return features


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络 (FPN)
    融合多尺度特征
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        
        # 输出卷积
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 从低到高的多尺度特征
        Returns:
            fpn_features: FPN融合后的特征
        """
        # 横向连接
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )
        
        # 输出卷积
        return [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]


class MultiGranularityBackbone(nn.Module):
    """
    多粒度骨干网络
    ResNet + FPN → 多粒度特征 (fine/medium/coarse)
    
    输出：
        - fine:   高分辨率特征 [B, C, H/4, W/4]
        - medium: 中分辨率特征 [B, C, H/8, W/8]
        - coarse: 低分辨率特征 [B, C, H/16, W/16]
    """
    def __init__(
        self,
        backbone_type: str = "resnet50",
        fpn_channels: int = 256,
        granularity_levels: List[str] = ["fine", "medium", "coarse"]
    ):
        super().__init__()
        self.granularity_levels = granularity_levels
        
        # 解析backbone配置
        layers = {"resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3]}.get(
            backbone_type, [3, 4, 6, 3]
        )
        
        # 基础骨干
        self.backbone = ResNetBackbone(layers=layers)
        
        # FPN - 输入通道对应 stride_8/16/32 的特征 (256, 512, 1024)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024],
            out_channels=fpn_channels
        )
        
        # 多粒度投影
        self.granularity_projs = nn.ModuleDict({
            level: nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
            for level in granularity_levels
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            mg_features: 多粒度特征字典
        """
        # 提取多尺度特征
        backbone_feats = self.backbone(x)
        
        # 按stride排序
        features_list = [
            backbone_feats[f"stride_{s}"] 
            for s in sorted([8, 16, 32])
        ]
        
        # FPN融合
        fpn_features = self.fpn(features_list)
        
        # 生成多粒度特征
        mg_features = {}
        for i, level in enumerate(self.granularity_levels):
            if i < len(fpn_features):
                feat = fpn_features[i]
                # 根据粒度调整分辨率
                if level == "coarse":
                    feat = F.adaptive_avg_pool2d(feat, (feat.shape[2] // 2, feat.shape[3] // 2))
                elif level == "fine":
                    feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
                
                mg_features[level] = self.granularity_projs[level](feat)
        
        return mg_features


def build_backbone(config: dict) -> nn.Module:
    """根据配置构建骨干网络"""
    return MultiGranularityBackbone(
        backbone_type=config.get("type", "resnet50"),
        fpn_channels=config.get("fpn_channels", 256),
        granularity_levels=config.get("granularity_levels", ["fine", "medium", "coarse"])
    )
