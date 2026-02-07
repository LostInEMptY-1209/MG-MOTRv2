"""
Demo script for MG-MOTRv2
MG-MOTRv2演示脚本 - 展示多粒度注意力机制的核心功能
"""

import torch
import torch.nn as nn
import argparse

from models import (
    MultiGranularityAttention,
    TemporalGranularityAttention,
    GranularityLevel,
    MultiGranularityBackbone,
)
from models.mg_motrv2 import MGMOTRv2
from configs.default_config import Config


def demo_granularity_level():
    """演示单层粒度注意力"""
    print("=" * 60)
    print("Demo: Granularity Level")
    print("=" * 60)
    
    # 创建粒度层
    layer = GranularityLevel(
        d_model=256,
        n_heads=8,
        granularity="medium",
        dropout=0.1
    )
    
    # 模拟输入
    batch_size, seq_len = 2, 100
    x = torch.randn(batch_size, seq_len, 256)
    
    # 前向传播
    output, aux = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary output shape: {aux.shape}")
    print(f"Granularity: {layer.granularity}, Stride: {layer.stride}")
    print()


def demo_multi_granularity_attention():
    """演示多粒度注意力"""
    print("=" * 60)
    print("Demo: Multi-Granularity Attention")
    print("=" * 60)
    
    # 创建多粒度注意力模块
    mg_attn = MultiGranularityAttention(
        d_model=256,
        n_heads=8,
        n_levels=3,
        fusion_type="adaptive",
        dropout=0.1
    )
    
    # 模拟多粒度输入
    batch_size = 2
    fine_feat = torch.randn(batch_size, 400, 256)    # 高分辨率
    medium_feat = torch.randn(batch_size, 100, 256)  # 中分辨率
    coarse_feat = torch.randn(batch_size, 25, 256)   # 低分辨率
    
    # 将不同粒度的特征统一尺寸（实际实现中可能需要更复杂的处理）
    # 这里简化为相同尺寸
    seq_len = 100
    fine_feat = fine_feat[:, :seq_len, :]
    medium_feat = medium_feat[:, :seq_len, :]
    coarse_feat = torch.nn.functional.interpolate(
        coarse_feat.transpose(1, 2), 
        size=seq_len, 
        mode='linear'
    ).transpose(1, 2)
    
    features = [fine_feat, medium_feat, coarse_feat]
    
    # 前向传播
    fused = mg_attn(features)
    
    print(f"Fine feature shape: {fine_feat.shape}")
    print(f"Medium feature shape: {medium_feat.shape}")
    print(f"Coarse feature shape: {coarse_feat.shape}")
    print(f"Fused output shape: {fused.shape}")
    print(f"Fusion type: {mg_attn.fusion_type}")
    print()


def demo_temporal_granularity_attention():
    """演示时序多粒度注意力"""
    print("=" * 60)
    print("Demo: Temporal Granularity Attention")
    print("=" * 60)
    
    # 创建时序多粒度注意力
    temporal_mg_attn = TemporalGranularityAttention(
        d_model=256,
        n_heads=8,
        n_frames=4,
        n_levels=3,
        dropout=0.1
    )
    
    # 模拟多帧多粒度输入
    batch_size = 2
    n_frames = 4
    seq_len = 100
    
    # 构建帧特征列表 [n_frames][n_levels][batch, seq, dim]
    frame_features = []
    for t in range(n_frames):
        frame_feats = [
            torch.randn(batch_size, seq_len, 256)
            for _ in range(3)
        ]
        frame_features.append(frame_feats)
    
    # 前向传播
    output = temporal_mg_attn(frame_features)
    
    print(f"Number of frames: {n_frames}")
    print(f"Number of granularity levels per frame: 3")
    print(f"Output shape: {output.shape}")
    print(f"Successfully fused temporal and multi-granularity features")
    print()


def demo_backbone():
    """演示多粒度骨干网络"""
    print("=" * 60)
    print("Demo: Multi-Granularity Backbone")
    print("=" * 60)
    
    # 创建骨干网络
    backbone = MultiGranularityBackbone(
        backbone_type="resnet50",
        fpn_channels=256,
        granularity_levels=["fine", "medium", "coarse"]
    )
    
    # 模拟输入图像
    batch_size = 2
    images = torch.randn(batch_size, 3, 800, 1333)
    
    # 前向传播
    mg_features = backbone(images)
    
    print(f"Input image shape: {images.shape}")
    print("Output multi-granularity features:")
    for level, feat in mg_features.items():
        print(f"  {level}: {feat.shape}")
    print()


def demo_full_model():
    """演示完整模型"""
    print("=" * 60)
    print("Demo: Full MG-MOTRv2 Model")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    config_dict = config.to_dict()
    
    # 创建模型
    model = MGMOTRv2(config_dict['model'])
    model.eval()
    
    # 模拟输入
    batch_size = 2
    images = torch.randn(batch_size, 3, 800, 1333)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Input image shape: {images.shape}")
    print("Model outputs:")
    print(f"  pred_logits: {outputs[0]['scores'].shape if isinstance(outputs, list) else 'N/A'}")
    print(f"  pred_boxes: {outputs[0]['boxes'].shape if isinstance(outputs, list) else 'N/A'}")
    print(f"  Number of detections (batch 0): {len(outputs[0]['scores']) if isinstance(outputs, list) else 'N/A'}")
    print()
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print()


def demo_model_structure():
    """展示模型结构"""
    print("=" * 60)
    print("Demo: Model Structure Overview")
    print("=" * 60)
    
    print("""
MG-MOTRv2 Architecture:
=======================

Input Image [B, 3, H, W]
    |
    v
+-----------------------------+
|  Multi-Granularity Backbone |  <- ResNet + FPN
|  - ResNet50/101             |
|  - FPN (Feature Pyramid)    |
+-----------------------------+
    |
    v
Multi-Granularity Features:
  - Fine   [B, C, H/4, W/4]    <- High resolution, detail info
  - Medium [B, C, H/8, W/8]    <- Balanced resolution
  - Coarse [B, C, H/16, W/16]  <- Low resolution, semantic info
    |
    v
+-----------------------------+
|  MG-DETR Head               |  <- Multi-Granularity DETR
|  - MG-Transformer Encoder   |     (Granularity fusion in encoder)
|  - Transformer Decoder      |
+-----------------------------+
    |
    v
+-----------------------------+
|  MG-Tracker                 |  <- Track management
|  - Query-based tracking     |
|  - Life cycle management    |
+-----------------------------+
    |
    v
Tracking Results: {track_id, bbox, score, ...}

Key Components:
===============
1. GranularityLevel: Single granularity attention module
2. MultiGranularityAttention: Fuses 3 granularity levels
3. TemporalGranularityAttention: Temporal + spatial multi-granularity
4. MultiGranularityBackbone: Extracts multi-scale features
5. MGTracker: Manages track life cycle
    """)
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser('MG-MOTRv2 Demo')
    parser.add_argument('--all', action='store_true', help='运行所有演示')
    parser.add_argument('--granularity', action='store_true', help='演示粒度层')
    parser.add_argument('--mg-attn', action='store_true', help='演示多粒度注意力')
    parser.add_argument('--temporal', action='store_true', help='演示时序注意力')
    parser.add_argument('--backbone', action='store_true', help='演示骨干网络')
    parser.add_argument('--model', action='store_true', help='演示完整模型')
    parser.add_argument('--structure', action='store_true', help='展示模型结构')
    args = parser.parse_args()
    
    # 如果没有指定特定演示，默认运行所有
    if not any([args.granularity, args.mg_attn, args.temporal, 
                args.backbone, args.model, args.structure]):
        args.all = True
    
    print("\n" + "=" * 60)
    print("MG-MOTRv2 Multi-Granularity Attention Demo")
    print("=" * 60 + "\n")
    
    if args.all or args.structure:
        demo_model_structure()
    
    if args.all or args.granularity:
        demo_granularity_level()
    
    if args.all or args.mg_attn:
        demo_multi_granularity_attention()
    
    if args.all or args.temporal:
        demo_temporal_granularity_attention()
    
    if args.all or args.backbone:
        demo_backbone()
    
    if args.all or args.model:
        demo_full_model()
    
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
