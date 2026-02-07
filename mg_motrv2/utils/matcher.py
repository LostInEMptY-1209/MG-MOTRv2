"""
Hungarian Matcher for MG-MOTRv2
匈牙利匹配器，用于DETR的预测与真值匹配
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器
    使用匈牙利算法将预测与真实目标进行最优匹配
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0
    ):
        """
        Args:
            cost_class: 分类代价的权重
            cost_bbox: L1边界框代价的权重
            cost_giou: GIoU边界框代价的权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "All costs cannot be 0"
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行匹配
        
        Args:
            outputs: 模型输出，包含:
                - "pred_logits": [batch_size, num_queries, num_classes]
                - "pred_boxes": [batch_size, num_queries, 4]
            targets: 真实目标列表，每个元素包含:
                - "labels": [num_target_boxes]
                - "boxes": [num_target_boxes, 4]
        Returns:
            indices: 匹配索引列表，每个元素为 (pred_indices, target_indices)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # 展平以计算代价矩阵
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # 拼接所有批次的目标
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 计算分类代价
        cost_class = -out_prob[:, tgt_ids]
        
        # 计算L1边界框代价
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # 计算GIoU边界框代价
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # 最终代价矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()
        
        # 确定每个批次的目标大小
        sizes = [len(v["boxes"]) for v in targets]
        
        # 使用匈牙利算法求解
        indices = []
        start_idx = 0
        for i, size in enumerate(sizes):
            if size == 0:
                # 如果没有目标，返回空匹配
                indices.append((torch.tensor([], dtype=torch.int64), 
                               torch.tensor([], dtype=torch.int64)))
                continue
            
            end_idx = start_idx + size
            c = C[i, :, start_idx:end_idx]
            
            # 匈牙利算法
            pred_indices, target_indices = linear_sum_assignment(c)
            
            indices.append((
                torch.as_tensor(pred_indices, dtype=torch.int64),
                torch.as_tensor(target_indices, dtype=torch.int64)
            ))
            start_idx = end_idx
        
        return indices


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """将中心点+宽高格式转换为左上角+右下角格式"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """将左上角+右下角格式转换为中心点+宽高格式"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算GIoU (Generalized Intersection over Union)
    
    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [M, 4] in xyxy format
    Returns:
        giou: [N, M] GIoU矩阵
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # 计算交集面积
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # 计算各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算并集面积
    union = area1[:, None] + area2 - inter
    
    # IoU
    iou = inter / (union + 1e-6)
    
    # 计算最小外接矩形
    lt_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_max = (rb_max - lt_min).clamp(min=0)
    area_max = wh_max[:, :, 0] * wh_max[:, :, 1]
    
    # GIoU
    giou = iou - (area_max - union) / (area_max + 1e-6)
    
    return giou


def build_matcher(config: dict) -> HungarianMatcher:
    """根据配置构建匹配器"""
    return HungarianMatcher(
        cost_class=config.get("cost_class", 1.0),
        cost_bbox=config.get("cost_bbox", 5.0),
        cost_giou=config.get("cost_giou", 2.0)
    )
