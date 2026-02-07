"""
MG-MOTRv2: Complete Model Integration
完整的多粒度多目标跟踪模型
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .backbone import MultiGranularityBackbone
from .mg_motr import MG_DETRHead, MGTracker


class MGMOTRv2(nn.Module):
    """
    完整的多粒度多目标跟踪模型
    
    Architecture:
        Input Image
            ↓
        Multi-Granularity Backbone (ResNet + FPN)
            ↓
        Multi-Granularity Features [fine, medium, coarse]
            ↓
        MG-DETR Head (Encoder-Decoder with MG-Attention)
            ↓
        Detection Outputs + Track Queries
            ↓
        MG-Tracker (Track Management)
            ↓
        Tracking Results
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 模型配置
        self.d_model = config.get("d_model", 256)
        self.num_queries = config.get("num_queries", 300)
        self.num_classes = config.get("num_classes", 1)
        self.n_frames = config.get("n_frames", 1)
        self.use_temporal = config.get("use_temporal", False)
        
        # 多粒度配置
        self.granularity_levels = config.get(
            "granularity_levels", ["fine", "medium", "coarse"]
        )
        self.n_granularity_levels = len(self.granularity_levels)
        
        # 1. 多粒度骨干网络
        backbone_config = config.get("backbone", {})
        backbone_config["fpn_channels"] = self.d_model
        backbone_config["granularity_levels"] = self.granularity_levels
        self.backbone = MultiGranularityBackbone(**backbone_config)
        
        # 2. 输入投影（将骨干特征投影到统一维度）
        self.input_proj = nn.ModuleDict({
            level: nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model, 1),
                nn.GroupNorm(32, self.d_model)
            )
            for level in self.granularity_levels
        })
        
        # 3. MG-DETR检测头
        detr_config = config.get("detr", {})
        detr_config.update({
            "d_model": self.d_model,
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "use_mg_attn": True,
            "n_granularity_levels": self.n_granularity_levels
        })
        self.detr_head = MG_DETRHead(**detr_config)
        
        # 4. 跟踪器
        tracker_config = config.get("tracker", {})
        tracker_config["d_model"] = self.d_model
        self.tracker = MGTracker(**tracker_config)
        
        # 5. 时序处理（可选）
        if self.use_temporal:
            self.temporal_fusion = TemporalFeatureFusion(
                d_model=self.d_model,
                n_frames=self.n_frames
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, p in self.named_parameters():
            if 'backbone' in name:
                continue  # 使用预训练权重
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def extract_features(
        self, 
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        提取多粒度特征
        
        Args:
            images: 输入图像 [B, 3, H, W]
        Returns:
            features: 多粒度特征字典
        """
        # 主干网络提取
        mg_features = self.backbone(images)
        
        # 投影到统一维度
        projected_features = {}
        for level, feat in mg_features.items():
            projected_features[level] = self.input_proj[level](feat)
        
        return projected_features
    
    def forward_single_frame(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        单帧前向传播
        
        Args:
            images: 输入图像 [B, 3, H, W]
            targets: 训练时的目标标注（可选）
        Returns:
            outputs: 包含检测和跟踪结果的字典
        """
        # 1. 提取多粒度特征
        mg_features = self.extract_features(images)
        
        # 准备DETR输入（使用中等粒度作为默认特征）
        default_level = "medium" if "medium" in mg_features else list(mg_features.keys())[0]
        default_feat = mg_features[default_level]
        
        # 准备多粒度特征列表
        mg_feat_list = [
            mg_features[level] for level in self.granularity_levels 
            if level in mg_features
        ]
        
        # 2. DETR前向传播
        detr_outputs = self.detr_head(
            default_feat,
            multi_granularity_features=mg_feat_list
        )
        
        # 3. 训练模式：直接返回损失
        if self.training and targets is not None:
            losses = self.compute_losses(detr_outputs, targets)
            return losses
        
        # 4. 推理模式：后处理并更新跟踪
        results = self.post_process(detr_outputs)
        
        return results
    
    def forward_sequence(
        self,
        images: torch.Tensor,
        targets: Optional[List[List[Dict]]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        序列前向传播（视频模式）
        
        Args:
            images: 输入图像序列 [B, T, 3, H, W]
            targets: 每帧的目标标注
        Returns:
            results: 每帧的跟踪结果列表
        """
        B, T, C, H, W = images.shape
        
        # 处理每帧
        all_results = []
        self.tracker.init_tracks()  # 重置跟踪器
        
        for t in range(T):
            frame = images[:, t]  # [B, 3, H, W]
            target = targets[t] if targets is not None else None
            
            result = self.forward_single_frame(frame, target)
            all_results.append(result)
        
        return all_results
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        模型前向传播入口
        
        Args:
            images: 输入图像 [B, 3, H, W] 或 [B, T, 3, H, W]
            targets: 目标标注
        Returns:
            outputs: 模型输出
        """
        if images.dim() == 5:
            # 视频序列模式
            return self.forward_sequence(images, targets)
        else:
            # 单帧模式
            return self.forward_single_frame(images, targets)
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失
        
        Args:
            outputs: DETR输出
            targets: 目标标注
        Returns:
            losses: 损失字典
        """
        # 这里简化实现，实际应使用匈牙利匹配
        losses = {}
        
        # 分类损失
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        
        # 占位符损失计算
        losses["loss_ce"] = torch.tensor(0.0, device=pred_logits.device)
        losses["loss_bbox"] = torch.tensor(0.0, device=pred_boxes.device)
        losses["loss_giou"] = torch.tensor(0.0, device=pred_boxes.device)
        
        return losses
    
    def post_process(
        self,
        outputs: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        后处理DETR输出
        
        Args:
            outputs: DETR输出
            threshold: 置信度阈值
        Returns:
            results: 处理后的检测结果列表
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        
        prob = pred_logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)
        
        results = []
        for i in range(pred_logits.shape[0]):
            # 筛选高置信度检测
            keep = scores[i] > threshold
            
            result = {
                "scores": scores[i][keep],
                "labels": labels[i][keep],
                "boxes": pred_boxes[i][keep]
            }
            results.append(result)
        
        return results


class TemporalFeatureFusion(nn.Module):
    """
    时序特征融合模块
    用于处理视频序列
    """
    def __init__(self, d_model: int, n_frames: int):
        super().__init__()
        self.d_model = d_model
        self.n_frames = n_frames
        
        # 时序注意力
        self.temporal_attn = nn.MultiheadAttention(
            d_model, 8, batch_first=True
        )
        
        # 位置编码
        self.temporal_pos = nn.Parameter(
            torch.randn(n_frames, d_model)
        )
        
        # 时序融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * n_frames, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, frame_features: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多帧特征
        
        Args:
            frame_features: 多帧特征列表，每个元素为 [B, N, C]
        Returns:
            fused: 融合后的特征 [B, N, C]
        """
        B, N, C = frame_features[0].shape
        
        # 堆叠时序维度
        feat_stack = torch.stack(frame_features, dim=1)  # [B, T, N, C]
        feat_stack = feat_stack.permute(0, 2, 1, 3)  # [B, N, T, C]
        feat_flat = feat_stack.reshape(B * N, self.n_frames, C)
        
        # 添加时序位置编码
        feat_pos = feat_flat + self.temporal_pos.unsqueeze(0)
        
        # 时序自注意力
        attn_out, _ = self.temporal_attn(feat_pos, feat_pos, feat_pos)
        
        # 融合
        attn_flat = attn_out.reshape(B, N, self.n_frames * C)
        fused = self.fusion_mlp(attn_flat)
        
        return fused


def build_model(config: dict) -> MGMOTRv2:
    """
    构建MG-MOTRv2模型
    
    Args:
        config: 模型配置字典
    Returns:
        model: 构建好的模型
    """
    return MGMOTRv2(config)
