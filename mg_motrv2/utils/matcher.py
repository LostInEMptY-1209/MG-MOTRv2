"""
Hungarian Matcher for MG-MOTRv2
匈牙利匹配器 - 预测与真值的最优匹配
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """中心点+宽高 → 左上角+右下角"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([
        cx - 0.5 * w, cy - 0.5 * h,
        cx + 0.5 * w, cy + 0.5 * h
    ], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算GIoU"""
    # 交集
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # 面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    
    iou = inter / (union + 1e-6)
    
    # 最小外接矩形
    lt_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_max = (rb_max - lt_min).clamp(min=0)
    area_max = wh_max[:, :, 0] * wh_max[:, :, 1]
    
    giou = iou - (area_max - union) / (area_max + 1e-6)
    return giou


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器
    使用匈牙利算法求解最优匹配
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: {"pred_logits": [B, N, C], "pred_boxes": [B, N, 4]}
            targets: 真值列表
        Returns:
            indices: 匹配索引列表
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # 展平
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 计算代价
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()
        
        # 匈牙利算法
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        start_idx = 0
        
        for size in sizes:
            if size == 0:
                indices.append((torch.tensor([], dtype=torch.int64), 
                               torch.tensor([], dtype=torch.int64)))
                continue
            
            end_idx = start_idx + size
            c = C[0, :, start_idx:end_idx] if batch_size == 1 else C[len(indices), :, :size]
            
            pred_indices, target_indices = linear_sum_assignment(c)
            indices.append((
                torch.as_tensor(pred_indices, dtype=torch.int64),
                torch.as_tensor(target_indices, dtype=torch.int64)
            ))
            start_idx = end_idx
        
        return indices


def build_matcher(config: dict) -> HungarianMatcher:
    """根据配置构建匹配器"""
    return HungarianMatcher(
        cost_class=config.get("cost_class", 1.0),
        cost_bbox=config.get("cost_bbox", 5.0),
        cost_giou=config.get("cost_giou", 2.0)
    )
