"""
MG-MOTRv2 Models - 模型模块导出
"""

from .mg_attention import (
    MultiGranularityAttention,
    TemporalGranularityAttention,
    GranularityLevel,
    build_mg_attention
)

from .mg_motr import (
    MG_DETRHead,
    MGTransformerEncoder,
    MGTransformerDecoder,
    MGTracker,
)

from .backbone import (
    MultiGranularityBackbone,
    build_backbone
)

from .mg_motrv2 import (
    MGMOTRv2,
    build_model
)

__all__ = [
    # 核心多粒度注意力
    "MultiGranularityAttention",
    "TemporalGranularityAttention",
    "GranularityLevel",
    "build_mg_attention",
    # DETR和跟踪
    "MG_DETRHead",
    "MGTransformerEncoder",
    "MGTransformerDecoder",
    "MGTracker",
    # 骨干网络
    "MultiGranularityBackbone",
    "build_backbone",
    # 完整模型
    "MGMOTRv2",
    "build_model",
]
