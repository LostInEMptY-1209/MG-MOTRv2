"""
Loss functions for MG-MOTRv2
损失函数 - DETR风格的集合预测损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    """
    集合损失函数（DETR风格）
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: List[str] = ["labels", "boxes"]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        
        # 类别权重（背景权重较低）
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int
    ) -> Dict[str, torch.Tensor]:
        """分类损失"""
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        return {'loss_ce': loss_ce}
    
    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int
    ) -> Dict[str, torch.Tensor]:
        """边界框损失"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        
        # L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        
        # GIoU损失
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / num_boxes
        
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
    
    def _get_src_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # 匹配
        indices = self.matcher(outputs_without_aux, targets)
        
        # 归一化因子
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, 
                                     device=next(iter(outputs.values())).device)
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
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'Loss {loss} not supported'
        return loss_map[loss](outputs, targets, indices, num_boxes)


def build_criterion(config: dict, matcher: HungarianMatcher) -> SetCriterion:
    """根据配置构建损失函数"""
    return SetCriterion(
        num_classes=config["model"]["num_classes"],
        matcher=matcher,
        weight_dict=config["train"]["loss_weights"],
        eos_coef=0.1,
        losses=["labels", "boxes"]
    )
