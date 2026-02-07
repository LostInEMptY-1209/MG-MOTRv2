"""
MG-MOTRv2: Multi-Granularity MOTR with enhanced attention mechanism
多粒度多目标跟踪Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .mg_attention import MultiGranularityAttention, TemporalGranularityAttention, build_mg_attention


class MG_DETRHead(nn.Module):
    """
    多粒度DETR检测头
    """
    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 1,  # 通常跟踪只关心前景/背景
        num_queries: int = 300,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        dropout: float = 0.1,
        use_mg_attn: bool = True,
        n_granularity_levels: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_mg_attn = use_mg_attn
        
        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # 编码器
        if use_mg_attn:
            self.encoder = MGTransformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_encoder_layers,
                n_granularity_levels=n_granularity_levels,
                dropout=dropout
            )
        else:
            self.encoder = StandardTransformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_encoder_layers,
                dropout=dropout
            )
        
        # 解码器
        self.decoder = MGTransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            dropout=dropout,
            use_mg_attn=use_mg_attn
        )
        
        # 输出头
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # (x, y, w, h)
        
    def forward(
        self,
        features: torch.Tensor,
        multi_granularity_features: Optional[List[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: 主干网络特征 [B, C, H, W]
            multi_granularity_features: 多粒度特征列表（可选）
            mask: 特征掩码
        Returns:
            outputs: 包含class和bbox预测的字典
        """
        B, C, H, W = features.shape
        
        # 展平特征 [B, C, H, W] -> [B, H*W, C]
        src = features.flatten(2).permute(0, 2, 1)
        
        # 生成位置编码
        pos_embed = self.pos_encoding(H * W, B).to(src.device)
        
        # 编码器
        if self.use_mg_attn and multi_granularity_features is not None:
            # 准备多粒度特征
            mg_features = [
                feat.flatten(2).permute(0, 2, 1) 
                for feat in multi_granularity_features
            ]
            memory = self.encoder(src, mg_features, pos_embed, mask)
        else:
            memory = self.encoder(src, pos_embed, mask)
        
        # 查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        tgt = torch.zeros_like(query_embed)
        
        # 解码器
        hs = self.decoder(tgt, memory, query_embed, pos_embed)
        
        # 输出
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "aux_outputs": [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        }


class MGTransformerEncoder(nn.Module):
    """
    多粒度Transformer编码器
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_granularity_levels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MGTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_granularity_levels=n_granularity_levels,
                dropout=dropout,
                use_mg_attn=(i >= n_layers // 2)  # 后半部分使用多粒度
            )
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        src: torch.Tensor,
        mg_features: List[torch.Tensor],
        pos_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, mg_features, pos_embed, mask)
        return self.norm(output)


class MGTransformerEncoderLayer(nn.Module):
    """
    多粒度Transformer编码器层
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_granularity_levels: int = 3,
        dropout: float = 0.1,
        use_mg_attn: bool = True
    ):
        super().__init__()
        self.use_mg_attn = use_mg_attn
        
        if use_mg_attn:
            self.mg_attn = MultiGranularityAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_levels=n_granularity_levels,
                dropout=dropout
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        src: torch.Tensor,
        mg_features: List[torch.Tensor],
        pos_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 添加位置编码
        q = src + pos_embed
        
        if self.use_mg_attn:
            # 使用多粒度注意力
            # 将当前特征与多粒度特征结合
            all_features = [src] + mg_features[:2]  # 限制为3个层级
            attn_out = self.mg_attn(all_features[:3])
            src = src + attn_out
        else:
            # 标准自注意力
            attn_out, _ = self.self_attn(q, q, src, key_padding_mask=mask)
            src = src + attn_out
        
        src = self.norm1(src)
        src = src + self.ffn(self.norm2(src))
        
        return src


class MGTransformerDecoder(nn.Module):
    """
    多粒度Transformer解码器
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        use_mg_attn: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MGTransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_mg_attn=use_mg_attn
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor
    ) -> torch.Tensor:
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, query_embed, pos_embed)
            intermediate.append(self.norm(output))
        
        return torch.stack(intermediate)


class MGTransformerDecoderLayer(nn.Module):
    """
    多粒度Transformer解码器层
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_mg_attn: bool = True
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor
    ) -> torch.Tensor:
        # 自注意力
        q = tgt + query_embed
        attn_out, _ = self.self_attn(q, q, tgt)
        tgt = tgt + attn_out
        tgt = self.norm1(tgt)
        
        # 交叉注意力
        q = tgt + query_embed
        k = memory + pos_embed
        attn_out, _ = self.cross_attn(q, k, memory)
        tgt = tgt + attn_out
        tgt = self.norm2(tgt)
        
        # FFN
        tgt = tgt + self.ffn(self.norm3(tgt))
        
        return tgt


class MGTracker(nn.Module):
    """
    多粒度跟踪器
    管理目标跟踪的生命周期和轨迹关联
    """
    def __init__(
        self,
        d_model: int = 256,
        max_tracks: int = 100,
        track_threshold: float = 0.5,
        missed_tolerance: int = 30
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tracks = max_tracks
        self.track_threshold = track_threshold
        self.missed_tolerance = missed_tolerance
        
        # 轨迹嵌入
        self.track_embed = nn.Embedding(max_tracks, d_model)
        
        # 轨迹更新网络
        self.track_update = nn.GRUCell(d_model, d_model)
        
        # 轨迹-检测匹配网络
        self.match_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 状态记录
        self.tracks = {}
        self.next_id = 0
        
    def init_tracks(self):
        """重置所有轨迹"""
        self.tracks = {}
        self.next_id = 0
        
    def update(
        self,
        detections: torch.Tensor,
        detection_features: torch.Tensor,
        frame_id: int
    ) -> List[Dict]:
        """
        更新跟踪状态
        
        Args:
            detections: 检测结果 [N, 4] (x, y, w, h)
            detection_features: 检测特征 [N, C]
            frame_id: 当前帧ID
        Returns:
            tracks: 当前活跃的跟踪目标列表
        """
        # 轨迹-检测匹配（简化版匈牙利匹配）
        if len(self.tracks) == 0:
            # 初始化新轨迹
            tracks = []
            for i, (det, feat) in enumerate(zip(detections, detection_features)):
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "id": track_id,
                    "bbox": det,
                    "feature": feat,
                    "frame_id": frame_id,
                    "missed": 0
                }
                tracks.append(self.tracks[track_id])
            return tracks
        
        # 计算匹配分数（简化实现）
        track_ids = list(self.tracks.keys())
        track_features = torch.stack([self.tracks[tid]["feature"] for tid in track_ids])
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(detection_features, track_features.t())
        
        # 简单贪心匹配（可替换为匈牙利算法）
        matched_det_indices = []
        matched_track_ids = []
        
        for i, det_feat in enumerate(detection_features):
            best_match = -1
            best_score = self.track_threshold
            for j, tid in enumerate(track_ids):
                if tid in matched_track_ids:
                    continue
                score = sim_matrix[i, j].item()
                if score > best_score:
                    best_score = score
                    best_match = tid
            
            if best_match >= 0:
                matched_det_indices.append(i)
                matched_track_ids.append(best_match)
                # 更新轨迹
                self.tracks[best_match]["bbox"] = detections[i]
                self.tracks[best_match]["feature"] = det_feat
                self.tracks[best_match]["frame_id"] = frame_id
                self.tracks[best_match]["missed"] = 0
        
        # 未匹配的检测 -> 新轨迹
        for i, (det, feat) in enumerate(zip(detections, detection_features)):
            if i not in matched_det_indices:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "id": track_id,
                    "bbox": det,
                    "feature": feat,
                    "frame_id": frame_id,
                    "missed": 0
                }
        
        # 未匹配的轨迹 -> 标记为missed
        active_tracks = []
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.tracks[tid]["missed"] += 1
            
            # 保留活跃轨迹
            if self.tracks[tid]["missed"] < self.missed_tolerance:
                active_tracks.append(self.tracks[tid])
            else:
                del self.tracks[tid]
        
        return active_tracks


# 辅助模块
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, length: int, batch_size: int) -> torch.Tensor:
        return self.pe[:length].unsqueeze(0).repeat(batch_size, 1, 1)


class MLP(nn.Module):
    """多层感知机"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class StandardTransformerEncoder(nn.Module):
    """标准Transformer编码器（备用）"""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
    def forward(
        self,
        src: torch.Tensor,
        pos_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder(src + pos_embed, src_key_padding_mask=mask)
