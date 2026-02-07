"""
MG-DETR Head for MG-MOTRv2
多粒度DETR检测头 - 基于Transformer的检测与跟踪

核心组件：
1. MGTransformerEncoder: 使用多粒度注意力的编码器
2. MGTransformerDecoder: 标准Transformer解码器
3. MGTracker: 轨迹管理器
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .mg_attention import MultiGranularityAttention
import torch.nn.functional as F
from mg_motrv2.models.ops.ms_deform_attn import MSDeformAttn


class MGTransformerEncoder(nn.Module):
    """
    多粒度Transformer编码器
    后半部分层使用多粒度注意力
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
                d_model, n_heads, n_granularity_levels, dropout,
                use_mg_attn=(i >= n_layers // 2)  # 后半部分使用MG-Attention
            )
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        src: torch.Tensor,
        mg_features: List[torch.Tensor],
        pos_embed: torch.Tensor
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, mg_features, pos_embed)
        return self.norm(output)



class MGTransformerEncoderLayer(nn.Module):
    """多粒度Transformer编码器层"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_granularity_levels: int,
        dropout: float,
        use_mg_attn: bool = True
    ):
        super().__init__()
        self.use_mg_attn = use_mg_attn
        
        if use_mg_attn:
            self.mg_attn = MultiGranularityAttention(
                d_model, n_heads, n_granularity_levels, dropout=dropout
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

    def chunked_self_attention(self, src, pos_embed, chunk_size=512):
        B, N, C = src.shape
        out = torch.zeros_like(src)

        for i in range(0, N, chunk_size):
            q = src[:, i:i + chunk_size] + pos_embed[:, i:i + chunk_size]
            k = src + pos_embed
            v = src

            attn_chunk, _ = self.self_attn(q, k, v)
            out[:, i:i + chunk_size] = attn_chunk

        return out
    def forward(
        self,
        src: torch.Tensor,
        mg_features: List[torch.Tensor],
        pos_embed: torch.Tensor
    ) -> torch.Tensor:
        if self.use_mg_attn:
            # 使用多粒度注意力
            # src: [B, N, C]
            all_features = [src] + mg_features[:2]

            target_len = src.shape[1]  # 以 src 的 token 数作为基准
            aligned_features = []

            for feat in all_features:
                # feat: [B, N_i, C]
                if feat.shape[1] != target_len:
                    # 在 token 维度上做插值对齐
                    feat = F.interpolate(
                        feat.permute(0, 2, 1),  # [B, C, N_i]
                        size=target_len,
                        mode="linear",
                        align_corners=False
                    ).permute(0, 2, 1)  # [B, N, C]

                aligned_features.append(feat)

            attn_out = self.mg_attn(aligned_features)
            src = src + attn_out
        else:
            attn_out = self.chunked_self_attention(
                src, pos_embed, chunk_size=512
            )
            src = src + attn_out

        src = self.norm1(src)
        src = src + self.ffn(self.norm2(src))
        
        return src

class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        self.cross_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # reference point predictor
        self.ref_point_head = nn.Linear(d_model, 2)

    def forward(
        self,
        tgt,                   # [B, Q, C]
        query_embed,           # [B, Q, C]
        multi_scale_features,  # list of [B, C, H, W]
        spatial_shapes,        # [n_levels, 2]
        level_start_index      # [n_levels]
    ):
        B, Q, C = tgt.shape
        n_levels = spatial_shapes.shape[0]

        # 1️⃣ self-attention (Q × Q, very small)
        q = tgt + query_embed
        attn_out, _ = self.self_attn(q, q, tgt)
        tgt = self.norm1(tgt + attn_out)

        # 2️⃣ reference points: [B, Q, n_levels, 2]
        reference_points = self.ref_point_head(tgt).sigmoid()
        reference_points = reference_points[:, :, None, :].repeat(
            1, 1, n_levels, 1
        )

        # 3️⃣ flatten multi-scale features
        flattened = []
        for feat in multi_scale_features:
            B, C, H, W = feat.shape
            flattened.append(feat.flatten(2).transpose(1, 2))

        input_flatten = torch.cat(flattened, dim=1)

        # 4️⃣ deformable cross-attention
        tgt2 = self.cross_attn(
            query=tgt,
            reference_points=reference_points,
            input_flatten=input_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        tgt = self.norm2(tgt + tgt2)

        # 5️⃣ FFN
        tgt = self.norm3(tgt + self.ffn(tgt))

        return tgt



class MGTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        n_levels=3
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_levels=n_levels
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt,
        query_embed,
        multi_scale_features,
        spatial_shapes,
        level_start_index
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                query_embed,
                multi_scale_features,
                spatial_shapes,
                level_start_index
            )
            intermediate.append(self.norm(output))

        return torch.stack(intermediate)


class MGTransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
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


class PositionEmbeddingSine(nn.Module):
    """
    2D sine-cosine positional encoding (DETR style)
    """
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, feat: torch.Tensor):
        # feat: [B, C, H, W]
        B, _, H, W = feat.shape
        device = feat.device

        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        dim_t = torch.arange(self.num_pos_feats, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        # ⬇⬇⬇ 这里是关键修正点 ⬇⬇⬇
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=-1
        ).flatten(2)

        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
            dim=-1
        ).flatten(2)

        # [H, W, C]
        pos = torch.cat((pos_y, pos_x), dim=2)

        # [B, C, H, W]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)

        # [B, HW, C]
        return pos.flatten(2).permute(0, 2, 1)



class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MG_DETRHead(nn.Module):
    """
    多粒度DETR检测头
    """
    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 1,
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
        
        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 位置编码
        self.pos_encoding = PositionEmbeddingSine(d_model // 2)
        
        # 编码器
        if use_mg_attn:
            self.encoder = MGTransformerEncoder(
                d_model, n_heads, n_encoder_layers, n_granularity_levels, dropout
            )
        else:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True),
                n_encoder_layers
            )
        
        # 解码器
        self.decoder = MGTransformerDecoder(d_model, n_heads, n_decoder_layers, dropout)
        
        # 输出头
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # (x, y, w, h)
        
        self.use_mg_attn = use_mg_attn
    
    def forward(
        self,
        features: torch.Tensor,
        multi_granularity_features: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: 主干网络特征 [B, C, H, W]
            multi_granularity_features: 多粒度特征列表（可选）
        Returns:
            outputs: 包含class和bbox预测的字典
        """
        B, C, H, W = features.shape
        
        # 展平特征 [B, C, H, W] -> [B, H*W, C]
        src = features.flatten(2).permute(0, 2, 1)
        
        # 生成位置编码
        pos_embed = self.pos_encoding(features).to(src.device)
        
        # 编码器
        multi_scale_features = multi_granularity_features

        spatial_shapes = torch.as_tensor(
            [(feat.shape[2], feat.shape[3]) for feat in multi_scale_features],
            dtype=torch.long,
            device=features.device
        )

        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1]
            )
        )

        # 查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(
            tgt,
            query_embed,
            multi_scale_features,
            spatial_shapes,
            level_start_index
        )

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


class MGTracker(nn.Module):
    """
    多粒度跟踪器
    管理目标跟踪的生命周期
    
    TODO: 可扩展功能：
    - 更精细的特征匹配
    - ReID特征学习
    - 卡尔曼滤波
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
        
        # 状态记录（非参数）
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
        更新跟踪状态（简化版贪心匹配）
        
        Args:
            detections: 检测结果 [N, 4]
            detection_features: 检测特征 [N, C]
            frame_id: 当前帧ID
        Returns:
            tracks: 活跃轨迹列表
        """
        # 初始化第一帧
        if len(self.tracks) == 0:
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
        
        # 计算匹配分数
        track_ids = list(self.tracks.keys())
        track_features = torch.stack([self.tracks[tid]["feature"] for tid in track_ids])
        
        sim_matrix = torch.mm(detection_features, track_features.t())
        
        # 贪心匹配
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
        
        # 保留活跃轨迹
        active_tracks = []
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.tracks[tid]["missed"] += 1
            
            if self.tracks[tid]["missed"] < self.missed_tolerance:
                active_tracks.append(self.tracks[tid])
            else:
                del self.tracks[tid]
        
        return active_tracks
