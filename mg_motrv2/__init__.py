"""
MG-MOTRv2: Multi-Granularity Multi-Object Tracking with Transformer
多粒度多目标跟踪Transformer

A research framework for multi-granularity attention mechanism in MOTR.
"""

__version__ = "0.1.0"
__author__ = "MG-MOTRv2 Research Team"

from .models import (
    MultiGranularityAttention,
    TemporalGranularityAttention,
    MultiGranularityBackbone,
    MG_DETRHead,
)

__all__ = [
    "MultiGranularityAttention",
    "TemporalGranularityAttention",
    "MultiGranularityBackbone",
    "MG_DETRHead",
]
