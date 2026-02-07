"""
Multi-Granularity Attention Mechanism for MOTRv2
多粒度注意力机制核心模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class GranularityLevel(nn.Module):
    """
    单一层级的粒度注意力模块
    支持不同空间分辨率的特征处理
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        granularity: str = "fine",  # "fine", "medium", "coarse"
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.granularity = granularity
        
        # 根据粒度设置下采样率
        self.stride = {"fine": 1, "medium": 2, "coarse": 4}[granularity]
        
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 跨粒度注意力（用于不同层级间交互）
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
        
        # 可选：粒度特定的投影层（预留扩展）
        self.granularity_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 输入特征 [B, H*W, C] 或 [B, N, C]
            memory: 来自其他粒度的特征（用于跨粒度交互）
            mask: 注意力掩码
        Returns:
            out: 输出特征
            aux: 辅助输出（用于层级间通信）
        """
        # 自注意力
        q = self.norm1(x)
        attn_out, _ = self.self_attn(q, q, q, attn_mask=mask)
        x = x + attn_out
        
        # 跨粒度注意力（如果提供了memory）
        if memory is not None:
            q = self.norm2(x)
            cross_out, _ = self.cross_attn(q, memory, memory)
            x = x + cross_out
        else:
            x = self.norm2(x)
        
        # 前馈网络
        x = x + self.ffn(self.norm3(x))
        
        # 粒度特定投影
        aux = self.granularity_proj(x)
        
        return x, aux


class MultiGranularityAttention(nn.Module):
    """
    多粒度注意力模块
    整合细粒度、中粒度和粗粒度三个层级的注意力
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 3,  # fine, medium, coarse
        fusion_type: str = "adaptive",  # "adaptive", "concat", "sum"
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.fusion_type = fusion_type
        
        # 创建不同粒度的注意力层
        self.granularity_levels = nn.ModuleList([
            GranularityLevel(
                d_model=d_model,
                n_heads=n_heads,
                granularity=["fine", "medium", "coarse"][i],
                dropout=dropout
            )
            for i in range(n_levels)
        ])
        
        # 粒度间交互门控
        self.level_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            for _ in range(n_levels - 1)
        ])
        
        # 特征融合
        if fusion_type == "adaptive":
            self.fusion_weights = nn.Sequential(
                nn.Linear(d_model * n_levels, n_levels),
                nn.Softmax(dim=-1)
            )
            self.output_proj = nn.Linear(d_model, d_model)
        elif fusion_type == "concat":
            self.output_proj = nn.Linear(d_model * n_levels, d_model)
        else:  # sum
            self.output_proj = nn.Identity()
            
    def forward(
        self,
        features: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            features: 不同粒度的输入特征列表，每个元素为 [B, N, C]
            masks: 可选的掩码列表
        Returns:
            fused: 融合后的特征 [B, N, C]
        """
        assert len(features) == self.n_levels, \
            f"Expected {self.n_levels} granularity levels, got {len(features)}"
        
        if masks is None:
            masks = [None] * self.n_levels
        
        # 处理每个粒度级别
        level_outputs = []
        prev_aux = None
        
        for i, (level, feat, mask) in enumerate(
            zip(self.granularity_levels, features, masks)
        ):
            # 粗粒度可以使用细粒度的辅助信息
            memory = prev_aux if i > 0 else None
            out, aux = level(feat, memory=memory, mask=mask)
            level_outputs.append(out)
            prev_aux = aux
        
        # 特征融合
        if self.fusion_type == "adaptive":
            # 拼接所有层级特征
            concat_feat = torch.cat(level_outputs, dim=-1)  # [B, N, C*n_levels]
            weights = self.fusion_weights(concat_feat)  # [B, N, n_levels]
            
            # 加权融合
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
    支持视频序列的多帧多粒度处理
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
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            dropout=dropout
        )
        
        # 时序注意力（跨帧）
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 时序融合
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * n_frames, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        frame_features: List[List[torch.Tensor]],  # [n_frames][n_levels][B, N, C]
        masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            frame_features: 多帧多粒度特征
            masks: 可选掩码
        Returns:
            output: 时序融合后的特征 [B, N, C]
        """
        n_frames = len(frame_features)
        batch_size = frame_features[0][0].shape[0]
        
        # 处理每帧的空间多粒度特征
        spatial_outputs = []
        for frame_feat in frame_features:
            out = self.spatial_mg_attn(frame_feat, masks)
            spatial_outputs.append(out)
        
        # 堆叠时序维度 [B, T, N, C]
        temporal_stack = torch.stack(spatial_outputs, dim=1)
        B, T, N, C = temporal_stack.shape
        
        # 重塑为 [B*N, T, C] 进行时序注意力
        temporal_flat = temporal_stack.view(B * N, T, C)
        temporal_out, _ = self.temporal_attn(
            temporal_flat, temporal_flat, temporal_flat
        )
        
        # 重塑回 [B, N, T, C] 并融合
        temporal_out = temporal_out.view(B, N, T, C)
        temporal_out = temporal_out.permute(0, 2, 1, 3)  # [B, T, N, C]
        
        # 时序融合：将多帧特征融合为单帧
        fused_frames = []
        for t in range(T):
            frame_feat = temporal_out[:, t, :, :]  # [B, N, C]
            # 可以添加帧间注意力机制（预留扩展）
            fused_frames.append(frame_feat)
        
        # 简单平均融合（可替换为更复杂的融合策略）
        output = sum(fused_frames) / len(fused_frames)
        output = self.norm(output)
        
        return output


# 便捷函数：创建标准配置的多粒度注意力
def build_mg_attention(config: dict) -> nn.Module:
    """
    根据配置构建多粒度注意力模块
    
    Args:
        config: 配置字典，包含以下键：
            - type: "spatial" 或 "temporal"
            - d_model: 特征维度
            - n_heads: 注意力头数
            - n_levels: 粒度级别数
            - n_frames: 帧数（仅时序模式）
            - dropout: dropout率
            - fusion_type: 融合类型
    """
    attn_type = config.get("type", "spatial")
    
    common_args = {
        "d_model": config.get("d_model", 256),
        "n_heads": config.get("n_heads", 8),
        "n_levels": config.get("n_levels", 3),
        "dropout": config.get("dropout", 0.1),
    }
    
    if attn_type == "temporal":
        return TemporalGranularityAttention(
            **common_args,
            n_frames=config.get("n_frames", 4)
        )
    else:
        return MultiGranularityAttention(
            **common_args,
            fusion_type=config.get("fusion_type", "adaptive")
        )
