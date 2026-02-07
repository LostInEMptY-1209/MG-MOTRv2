"""
MG-MOTRv2: Multi-Granularity Multi-Object Tracking with Transformer
多粒度多目标跟踪Transformer研究框架

快速开始：
    from mg_motrv2.models import MultiGranularityAttention, build_model
    from mg_motrv2.configs import get_base_config
    
    config = get_base_config()
    model = build_model(config.to_dict()['model'])

核心模块：
    - models.mg_attention: 多粒度注意力机制
    - models.backbone: 多粒度骨干网络
    - models.mg_motr: MG-DETR检测头
    - models.mg_motrv2: 完整模型
    
扩展点：
    - 自定义粒度级别（mg_attention.GranularityLevel）
    - 自定义融合策略（mg_attention.MultiGranularityAttention）
    - 替换骨干网络（backbone.MultiGranularityBackbone）
"""

__version__ = "0.1.0"

from .models import (
    MultiGranularityAttention,
    TemporalGranularityAttention,
    MultiGranularityBackbone,
    MG_DETRHead,
    build_model,
)

__all__ = [
    "MultiGranularityAttention",
    "TemporalGranularityAttention",
    "MultiGranularityBackbone",
    "MG_DETRHead",
    "build_model",
]
