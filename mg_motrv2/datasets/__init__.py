"""
Dataset modules for MG-MOTRv2
MOT17 数据集加载
"""

from .mot17 import MOT17Dataset, build_mot17_dataloader

__all__ = [
    "MOT17Dataset",
    "build_mot17_dataloader",
]
