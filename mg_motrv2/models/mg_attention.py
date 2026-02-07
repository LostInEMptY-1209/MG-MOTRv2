"""
Multi-Granularity Attention Mechanism for MOTRv2
多粒度注意力机制 - 核心模块

核心创新：
1. 多粒度特征处理（细/中/粗粒度）
2. 自适应特征融合
3. 时序建模支持（可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class GranularityLevel(nn.Module):
    """
    单一层级的粒度注意力模块
    处理特定空间分辨率的特征
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        granularity: str = "fine",
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.granularity = granularity
        
        # 粒度对应的空间下采样率
        self.stride = {"fine": 1, "medium": 2, "coarse": 4}[granularity]
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 跨粒度注意力（层级间信息交互）
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, N, C]
            memory: 来自其他粒度的特征（可选）
        Returns:
            out: 输出特征 [B, N, C]
        """
        # 自注意力
        q = self.norm1(x)
        attn_out, _ = self.self_attn(q, q, q)
        x = x + attn_out
        
        # 跨粒度注意力
        if memory is not None:
            q = self.norm2(x)
            cross_out, _ = self.cross_attn(q, memory, memory)
            x = x + cross_out
        else:
            x = self.norm2(x)
        
        # 前馈网络
        x = x + self.ffn(self.norm3(x))
        
        return x


class MultiGranularityAttention(nn.Module):
    """
    多粒度注意力模块 - 核心组件
    整合细粒度、中粒度、粗粒度三个层级的特征
    
    架构：
        Input: [fine_feat, medium_feat, coarse_feat]
            ↓
        ┌─────────────────────────────────────┐
        │  GranularityLevel (fine)            │
        │  GranularityLevel (medium) ← fine   │
        │  GranularityLevel (coarse) ← medium │
        └─────────────────────────────────────┘
            ↓
        Fusion (adaptive/concat/sum)
            ↓
        Output: fused_feature
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 3,
        fusion_type: str = "adaptive",
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.fusion_type = fusion_type
        
        # 创建不同粒度的注意力层
        granularities = ["fine", "medium", "coarse"][:n_levels]
        self.granularity_levels = nn.ModuleList([
            GranularityLevel(d_model, n_heads, g, dropout)
            for g in granularities
        ])
        
        # 特征融合模块
        if fusion_type == "adaptive":
            # 自适应权重融合
            self.fusion_weights = nn.Sequential(
                nn.Linear(d_model * n_levels, n_levels),
                nn.Softmax(dim=-1)
            )
            self.output_proj = nn.Linear(d_model, d_model)
        elif fusion_type == "concat":
            self.output_proj = nn.Linear(d_model * n_levels, d_model)
        else:  # sum
            self.output_proj = nn.Identity()
            
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 多粒度特征列表 [B, N, C]，共n_levels个
        Returns:
            fused: 融合后的特征 [B, N, C]
        """
        assert len(features) == self.n_levels
        
        # 逐层处理，粗粒度可访问细粒度的信息
        level_outputs = []
        prev_feat = None
        
        for level, feat in zip(self.granularity_levels, features):
            out = level(feat, memory=prev_feat)
            level_outputs.append(out)
            prev_feat = out
        
        # 特征融合
        if self.fusion_type == "adaptive":
            concat_feat = torch.cat(level_outputs, dim=-1)
            weights = self.fusion_weights(concat_feat)  # [B, N, n_levels]
            
            fused = sum(
                w.unsqueeze(-1) * feat
                for w, feat in zip(weights.unbind(-1), level_outputs)
            )
            fused = self.output_proj(fused)
            
        elif self.fusion_type == "concat":
            fused = torch.cat(level_outputs, dim=-1)
            fused = self.output_proj(fused)
        else:  # sum
            fused = sum(level_outputs)
            
        return fused


class TemporalGranularityAttention(nn.Module):
    """
    时序多粒度注意力模块
    支持视频序列的多帧处理（可选扩展）
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_frames: int = 4,
        n_levels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_frames = n_frames
        
        # 空间多粒度注意力
        self.spatial_mg_attn = MultiGranularityAttention(
            d_model, n_heads, n_levels, dropout=dropout
        )
        
        # 时序注意力
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, frame_features: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            frame_features: [n_frames][n_levels][B, N, C]
        Returns:
            output: 时序融合特征 [B, N, C]
        """
        # 处理每帧的空间特征
        spatial_outputs = [
            self.spatial_mg_attn(feats) 
            for feats in frame_features
        ]
        
        # 时序注意力 [B, T, N, C]
        temporal_stack = torch.stack(spatial_outputs, dim=1)
        B, T, N, C = temporal_stack.shape
        
        # 重塑为 [B*N, T, C]
        temporal_flat = temporal_stack.view(B * N, T, C)
        temporal_out, _ = self.temporal_attn(
            temporal_flat, temporal_flat, temporal_flat
        )
        
        # 平均池化时序维度
        temporal_out = temporal_out.view(B, N, T, C)
        output = temporal_out.mean(dim=2)  # [B, N, C]
        
        return self.norm(output)


def build_mg_attention(config: dict) -> nn.Module:
    """根据配置构建多粒度注意力模块"""
    attn_type = config.get("type", "spatial")
    common_args = {
        "d_model": config.get("d_model", 256),
        "n_heads": config.get("n_heads", 8),
        "n_levels": config.get("n_levels", 3),
        "dropout": config.get("dropout", 0.1),
    }
    
    if attn_type == "temporal":
        return TemporalGranularityAttention(
            **common_args, n_frames=config.get("n_frames", 4)
        )
    else:
        return MultiGranularityAttention(
            **common_args, fusion_type=config.get("fusion_type", "adaptive")
        )
