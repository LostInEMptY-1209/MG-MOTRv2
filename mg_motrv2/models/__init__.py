"""
MG-MOTRv2 Models
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
    ResNetBackbone,
    FeaturePyramidNetwork,
    build_backbone
)

__all__ = [
    # Attention modules
    "MultiGranularityAttention",
    "TemporalGranularityAttention", 
    "GranularityLevel",
    "build_mg_attention",
    # MOTR modules
    "MG_DETRHead",
    "MGTransformerEncoder",
    "MGTransformerDecoder",
    "MGTracker",
    # Backbone modules
    "MultiGranularityBackbone",
    "ResNetBackbone",
    "FeaturePyramidNetwork",
    "build_backbone",
]
