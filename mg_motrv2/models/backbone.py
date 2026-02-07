"""
Backbone networks for MG-MOTRv2
支持多粒度特征提取的骨干网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


class ResNetBackbone(nn.Module):
    """
    ResNet骨干网络，支持多尺度特征输出
    """
    def __init__(
        self,
        layers: List[int] = [3, 4, 6, 3],  # ResNet50 default
        block_type: str = "bottleneck",
        in_channels: int = 3,
        base_channels: int = 64,
        output_strides: List[int] = [8, 16, 32],  # 多粒度输出步长
        frozen_stages: int = 1
    ):
        super().__init__()
        self.output_strides = sorted(output_strides)
        self.frozen_stages = frozen_stages
        
        # 初始卷积
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, kernel_size=7, 
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建ResNet阶段
        self.in_channels = base_channels
        self.stages = nn.ModuleList()
        
        strides = [1, 2, 2, 2]
        channels = [base_channels * 4, base_channels * 8, base_channels * 16, base_channels * 32]
        
        for i, (num_blocks, stride, out_ch) in enumerate(zip(layers, strides, channels)):
            stage = self._make_layer(
                out_ch, num_blocks, stride=stride,
                block_type=block_type
            )
            self.stages.append(stage)
        
        # 冻结早期层
        self._freeze_stages()
        
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        block_type: str = "bottleneck"
    ) -> nn.Module:
        """构建ResNet阶段"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        blocks = []
        if block_type == "bottleneck":
            blocks.append(Bottleneck(self.in_channels, out_channels // 4, stride, downsample))
            self.in_channels = out_channels
            for _ in range(1, num_blocks):
                blocks.append(Bottleneck(self.in_channels, out_channels // 4))
        else:
            blocks.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*blocks)
    
    def _freeze_stages(self):
        """冻结早期层"""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in [self.conv1.parameters(), self.bn1.parameters()]:
                for p in param:
                    p.requires_grad = False
        
        for i in range(1, self.frozen_stages + 1):
            if i <= len(self.stages):
                self.stages[i-1].eval()
                for param in self.stages[i-1].parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回多尺度特征
        
        Returns:
            features: 字典，键为步长，值为对应尺度的特征
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        features = {}
        current_stride = 4  # 经过conv1和maxpool后的步长
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            current_stride *= 2
            
            if current_stride in self.output_strides:
                features[f"stride_{current_stride}"] = x
        
        return features


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络 (FPN)
    用于融合多尺度特征
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        use_extra_levels: bool = True
    ):
        super().__init__()
        self.use_extra_levels = use_extra_levels
        
        # 横向连接
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            )
        
        # 输出卷积
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels)
                )
            )
        
        # 额外的下采样层
        if use_extra_levels:
            self.extra_downsamples = nn.ModuleList()
            for _ in range(2):  # 添加P6, P7
                self.extra_downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True)
                    )
                )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 从低到高的多尺度特征
        Returns:
            fpn_features: FPN融合后的特征列表
        """
        # 横向连接
        laterals = [
            lateral_conv(f) 
            for lateral_conv, f in zip(self.lateral_convs, features)
        ]
        
        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )
        
        # 输出卷积
        fpn_features = [
            fpn_conv(lat) 
            for fpn_conv, lat in zip(self.fpn_convs, laterals)
        ]
        
        # 额外层级
        if self.use_extra_levels:
            last_feature = fpn_features[-1]
            for downsample in self.extra_downsamples:
                last_feature = downsample(last_feature)
                fpn_features.append(last_feature)
        
        return fpn_features


class MultiGranularityBackbone(nn.Module):
    """
    多粒度骨干网络
    整合ResNet + FPN，输出多粒度特征
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
        if backbone_type == "resnet50":
            backbone_layers = [3, 4, 6, 3]
        elif backbone_type == "resnet101":
            backbone_layers = [3, 4, 23, 3]
        else:
            backbone_layers = [2, 2, 2, 2]  # resnet18
        
        # 基础骨干
        self.backbone = ResNetBackbone(
            layers=backbone_layers,
            output_strides=[8, 16, 32]
        )
        
        # 获取各阶段通道数
        if "bottleneck" in str(backbone_layers):
            in_channels = [512, 1024, 2048]
        else:
            in_channels = [128, 256, 512]
        
        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=fpn_channels
        )
        
        # 多粒度投影
        self.granularity_projections = nn.ModuleDict({
            level: nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            )
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
        backbone_features = self.backbone(x)
        
        # 从低到高排序特征
        features_list = [
            backbone_features[f"stride_{s}"] 
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
                
                mg_features[level] = self.granularity_projections[level](feat)
        
        return mg_features


# ResNet基础模块
class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3, stride, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(
            mid_channels, mid_channels * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


def build_backbone(config: dict) -> nn.Module:
    """
    根据配置构建骨干网络
    
    Args:
        config: 配置字典
    """
    backbone_type = config.get("type", "resnet50")
    fpn_channels = config.get("fpn_channels", 256)
    granularity_levels = config.get("granularity_levels", ["fine", "medium", "coarse"])
    
    return MultiGranularityBackbone(
        backbone_type=backbone_type,
        fpn_channels=fpn_channels,
        granularity_levels=granularity_levels
    )
