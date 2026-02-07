"""
Loss functions for MG-MOTRv2
MG-MOTRv2损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    """
    集合损失函数
    用于DETR风格的集合预测
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: List[str] = ["labels", "boxes"]
    ):
        """
        Args:
            num_classes: 类别数
            matcher: 匈牙利匹配器
            weight_dict: 各损失的权重字典
            eos_coef: 背景类别的相对权重
            losses: 要计算的损失类型列表
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # 类别权重（背景类别权重较低）
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int
    ) -> Dict[str, torch.Tensor]:
        """分类损失"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        # 获取匹配的索引
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        # 计算交叉熵损失
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {'loss_ce': loss_ce}
        
        return losses
    
    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int
    ) -> Dict[str, torch.Tensor]:
        """边界框损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        
        # L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # GIoU损失
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def _get_src_permutation_idx(
        self,
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取源索引的排列"""
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(
        self,
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取目标索引的排列"""
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            outputs: 模型输出
            targets: 真实目标
        Returns:
            losses: 损失字典
        """
        # 移除aux_outputs以简化
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # 获取匹配
        indices = self.matcher(outputs_without_aux, targets)
        
        # 计算目标框总数（用于归一化）
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_boxes = torch.clamp(num_boxes, min=1.0).item()
        
        # 计算各项损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        return losses
    
    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int
    ) -> Dict[str, torch.Tensor]:
        """根据损失类型获取对应的损失"""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'Loss {loss} not supported'
        return loss_map[loss](outputs, targets, indices, num_boxes)


class TrackingLoss(nn.Module):
    """
    跟踪专用损失
    包括ReID损失、时序一致性损失等
    """
    
    def __init__(
        self,
        d_model: int = 256,
        use_reid: bool = True,
        use_temporal: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.use_reid = use_reid
        self.use_temporal = use_temporal
        
        if use_reid:
            # ReID对比损失
            self.reid_loss = ReIDLoss(d_model)
        
        if use_temporal:
            # 时序一致性损失
            self.temporal_loss = TemporalConsistencyLoss()
    
    def forward(
        self,
        track_features: torch.Tensor,
        track_ids: torch.Tensor,
        prev_features: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            track_features: 跟踪特征 [N, C]
            track_ids: 轨迹ID [N]
            prev_features: 历史特征（可选）
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        if self.use_reid:
            losses['loss_reid'] = self.reid_loss(track_features, track_ids)
        
        if self.use_temporal and prev_features is not None:
            losses['loss_temporal'] = self.temporal_loss(
                track_features, prev_features
            )
        
        return losses


class ReIDLoss(nn.Module):
    """ReID对比损失（简化版）"""
    
    def __init__(self, d_model: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Linear(d_model, 128)
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [N, C]
            labels: [N] 轨迹ID
        """
        # 投影到ReID空间
        embeddings = F.normalize(self.projection(features), dim=1)
        
        # 计算相似度矩阵
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # 创建正样本掩码
        labels = labels.unsqueeze(0)
        mask = (labels == labels.t()).float()
        
        # 移除对角线（自身）
        mask.fill_diagonal_(0)
        
        # 对比损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只计算正样本的损失
        loss = -(mask * log_prob).sum() / (mask.sum() + 1e-8)
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """时序一致性损失"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        curr_features: torch.Tensor,
        prev_features: torch.Tensor
    ) -> torch.Tensor:
        """
        鼓励相邻帧特征的一致性
        
        Args:
            curr_features: 当前帧特征 [N, C]
            prev_features: 前一帧特征 [N, C]
        """
        # L2距离
        distance = F.mse_loss(curr_features, prev_features, reduction='mean')
        return distance


def build_criterion(config: dict, matcher: HungarianMatcher) -> SetCriterion:
    """根据配置构建损失函数"""
    return SetCriterion(
        num_classes=config["model"]["num_classes"],
        matcher=matcher,
        weight_dict=config["train"]["loss_weights"],
        eos_coef=0.1,
        losses=["labels", "boxes"]
    )
