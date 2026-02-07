"""
MG-MOTRv2: Complete Model Integration
完整的多粒度多目标跟踪模型

数据流：
    Input Image [B, 3, H, W]
        ↓
    Multi-Granularity Backbone (ResNet + FPN)
        ↓
    Multi-Granularity Features {fine, medium, coarse}
        ↓
    MG-DETR Head (Encoder-Decoder with MG-Attention)
        ↓
    Detection Outputs + Track Queries
        ↓
    MG-Tracker
        ↓
    Tracking Results

扩展点：
1. 可替换骨干网络（backbone.py）
2. 可自定义多粒度注意力机制（mg_attention.py）
3. 可扩展时序建模（TemporalGranularityAttention）
4. 可集成ReID（TrackingLoss）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .backbone import MultiGranularityBackbone
from .mg_motr import MG_DETRHead, MGTracker


class MGMOTRv2(nn.Module):
    """完整的多粒度多目标跟踪模型"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 基础配置
        self.d_model = config.get("d_model", 256)
        self.num_queries = config.get("num_queries", 300)
        self.num_classes = config.get("num_classes", 1)
        self.granularity_levels = config.get("granularity_levels", ["fine", "medium", "coarse"])
        
        # 1. 多粒度骨干网络
        backbone_config = config.get("backbone", {})
        backbone_config["fpn_channels"] = self.d_model
        backbone_config["granularity_levels"] = self.granularity_levels
        # 映射 'type' 到 'backbone_type'
        if "type" in backbone_config:
            backbone_config["backbone_type"] = backbone_config.pop("type")
        self.backbone = MultiGranularityBackbone(**backbone_config)
        
        # 2. 输入投影
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
            "n_granularity_levels": len(self.granularity_levels)
        })
        self.detr_head = MG_DETRHead(**detr_config)
        
        # 4. 跟踪器
        tracker_config = config.get("tracker", {})
        tracker_config["d_model"] = self.d_model
        self.tracker = MGTracker(**tracker_config)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, p in self.named_parameters():
            if 'backbone' in name:
                continue  # 使用预训练权重
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取多粒度特征"""
        mg_features = self.backbone(images)
        
        # 投影到统一维度
        projected = {
            level: self.input_proj[level](feat)
            for level, feat in mg_features.items()
        }
        
        return projected
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        模型前向传播
        
        Args:
            images: 输入图像 [B, 3, H, W]
            targets: 训练目标（可选）
        Returns:
            outputs: 模型输出
        """
        # 1. 提取多粒度特征
        mg_features = self.extract_features(images)
        
        # 准备DETR输入（使用中等粒度作为默认）
        default_level = "medium" if "medium" in mg_features else list(mg_features.keys())[0]
        default_feat = mg_features[default_level]
        
        # 准备多粒度特征列表
        mg_feat_list = [
            mg_features[level] for level in self.granularity_levels 
            if level in mg_features
        ]
        
        # 2. DETR前向传播
        detr_outputs = self.detr_head(default_feat, mg_feat_list)
        
        # 3. 训练模式：返回损失占位符
        if self.training and targets is not None:
            return self._compute_losses(detr_outputs, targets)
        
        # 4. 推理模式：后处理
        return self._post_process(detr_outputs)
    
    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """计算训练损失（占位符）"""
        # TODO: 实现完整的损失计算（使用matcher和criterion）
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        
        return {
            "loss_ce": torch.tensor(0.0, device=pred_logits.device),
            "loss_bbox": torch.tensor(0.0, device=pred_boxes.device),
            "loss_giou": torch.tensor(0.0, device=pred_boxes.device),
        }
    
    def _post_process(
        self,
        outputs: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> List[Dict]:
        """后处理DETR输出"""
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        
        prob = pred_logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)
        
        results = []
        for i in range(pred_logits.shape[0]):
            keep = scores[i] > threshold
            
            results.append({
                "scores": scores[i][keep],
                "labels": labels[i][keep],
                "boxes": pred_boxes[i][keep]
            })
        
        return results


def build_model(config: dict) -> MGMOTRv2:
    """构建MG-MOTRv2模型"""
    return MGMOTRv2(config)
