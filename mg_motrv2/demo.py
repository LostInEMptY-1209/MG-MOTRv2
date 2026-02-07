"""
Demo script for MG-MOTRv2
演示脚本 - 展示多粒度注意力机制的核心功能

使用方法：
    python -m mg_motrv2.demo --all       # 运行所有演示
    python -m mg_motrv2.demo --mg-attn   # 仅运行多粒度注意力演示
"""

import torch
import argparse

from mg_motrv2.models import (
    MultiGranularityAttention,
    TemporalGranularityAttention,
    GranularityLevel,
    MultiGranularityBackbone,
)
from mg_motrv2.models.mg_motrv2 import MGMOTRv2
from mg_motrv2.configs.default_config import Config


def demo_granularity_level():
    """演示单层粒度注意力"""
    print("\n" + "=" * 60)
    print("Demo: Granularity Level")
    print("=" * 60)
    
    layer = GranularityLevel(d_model=256, n_heads=8, granularity="medium")
    x = torch.randn(2, 100, 256)
    out = layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Granularity: {layer.granularity}, Stride: {layer.stride}")


def demo_multi_granularity_attention():
    """演示多粒度注意力"""
    print("\n" + "=" * 60)
    print("Demo: Multi-Granularity Attention")
    print("=" * 60)
    
    mg_attn = MultiGranularityAttention(
        d_model=256, n_heads=8, n_levels=3, fusion_type="adaptive"
    )
    
    # 模拟多粒度输入（已统一尺寸）
    batch_size, seq_len = 2, 100
    features = [torch.randn(batch_size, seq_len, 256) for _ in range(3)]
    
    fused = mg_attn(features)
    
    print(f"Number of levels: {mg_attn.n_levels}")
    print(f"Fusion type: {mg_attn.fusion_type}")
    print(f"Fused output shape: {fused.shape}")


def demo_temporal_granularity_attention():
    """演示时序多粒度注意力"""
    print("\n" + "=" * 60)
    print("Demo: Temporal Granularity Attention")
    print("=" * 60)
    
    temporal_attn = TemporalGranularityAttention(
        d_model=256, n_heads=8, n_frames=4, n_levels=3
    )
    
    # 模拟多帧多粒度输入
    batch_size, seq_len = 2, 100
    frame_features = [
        [torch.randn(batch_size, seq_len, 256) for _ in range(3)]
        for _ in range(4)
    ]
    
    output = temporal_attn(frame_features)
    
    print(f"Number of frames: 4")
    print(f"Output shape: {output.shape}")


def demo_backbone():
    """演示多粒度骨干网络"""
    print("\n" + "=" * 60)
    print("Demo: Multi-Granularity Backbone")
    print("=" * 60)
    
    backbone = MultiGranularityBackbone(
        backbone_type="resnet50",
        fpn_channels=256,
        granularity_levels=["fine", "medium", "coarse"]
    )
    
    images = torch.randn(2, 3, 800, 1333)
    mg_features = backbone(images)
    
    print(f"Input image shape: {images.shape}")
    print("Output multi-granularity features:")
    for level, feat in mg_features.items():
        print(f"  {level}: {feat.shape}")


def demo_full_model():
    """演示完整模型"""
    print("\n" + "=" * 60)
    print("Demo: Full MG-MOTRv2 Model")
    print("=" * 60)
    
    config = Config()
    config_dict = config.to_dict()
    
    # 使用较小配置
    config_dict['model']['num_queries'] = 10
    config_dict['model']['detr']['n_encoder_layers'] = 2
    config_dict['model']['detr']['n_decoder_layers'] = 2
    
    model = MGMOTRv2(config_dict['model'])
    model.eval()
    
    images = torch.randn(1, 3, 400, 600)
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Input image shape: {images.shape}")
    print(f"Number of detections: {len(outputs[0]['scores'])}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")


def demo_model_structure():
    """展示模型结构"""
    print("\n" + "=" * 60)
    print("MG-MOTRv2 Architecture")
    print("=" * 60)
    print("""
Input Image [B, 3, H, W]
    ↓
+-----------------------------+
|  Multi-Granularity Backbone |
|  - ResNet50/101             |
|  - FPN                      |
+-----------------------------+
    ↓
Multi-Granularity Features:
  - Fine   [B, C, H/4, W/4]
  - Medium [B, C, H/8, W/8]
  - Coarse [B, C, H/16, W/16]
    ↓
+-----------------------------+
|  MG-DETR Head               |
|  - MG-Transformer Encoder   |
|  - Transformer Decoder      |
+-----------------------------+
    ↓
Detection Outputs
    ↓
+-----------------------------+
|  MG-Tracker                 |
+-----------------------------+
    ↓
Tracking Results

Key Components:
1. GranularityLevel: 单层粒度注意力
2. MultiGranularityAttention: 多粒度特征融合
3. TemporalGranularityAttention: 时序扩展
4. MultiGranularityBackbone: 多尺度特征提取
    """)


def main():
    """主函数"""
    parser = argparse.ArgumentParser('MG-MOTRv2 Demo')
    parser.add_argument('--all', action='store_true', help='运行所有演示')
    parser.add_argument('--granularity', action='store_true', help='粒度层演示')
    parser.add_argument('--mg-attn', action='store_true', help='多粒度注意力演示')
    parser.add_argument('--temporal', action='store_true', help='时序注意力演示')
    parser.add_argument('--backbone', action='store_true', help='骨干网络演示')
    parser.add_argument('--model', action='store_true', help='完整模型演示')
    parser.add_argument('--structure', action='store_true', help='模型结构说明')
    args = parser.parse_args()
    
    # 默认运行所有
    if not any([args.granularity, args.mg_attn, args.temporal, 
                args.backbone, args.model, args.structure]):
        args.all = True
    
    print("\n" + "=" * 60)
    print("MG-MOTRv2 Multi-Granularity Attention Demo")
    print("=" * 60)
    
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
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
