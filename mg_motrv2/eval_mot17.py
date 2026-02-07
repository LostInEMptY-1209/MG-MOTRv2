"""
MOT17 Evaluation Script
MOT17评估脚本 - 计算MOTA, IDF1等跟踪指标

使用方法：
    python -m mg_motrv2.eval_mot17 --checkpoint ./outputs/checkpoint_best.pth --dataset_root ./data/MOT17
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import numpy as np
import cv2

from .models.mg_motrv2 import build_model
from .configs.default_config import Config
from .datasets.mot17 import MOT17Transform


def get_args_parser():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser('MG-MOTRv2 MOT17 Evaluation')
    
    parser.add_argument('--checkpoint', required=True, type=str, help='模型检查点路径')
    parser.add_argument('--dataset_root', default='./data/MOT17', type=str, help='MOT17数据集根目录')
    parser.add_argument('--output_dir', default='./eval_results', type=str, help='输出目录')
    parser.add_argument('--split', default='train', type=str, choices=['train', 'test'], help='评估数据集')
    
    # 模型配置（会从检查点加载，也可覆盖）
    parser.add_argument('--conf_threshold', default=0.5, type=float, help='置信度阈值')
    parser.add_argument('--device', default='cuda', type=str, help='设备')
    
    return parser


class Track:
    """跟踪目标"""
    def __init__(self, track_id, bbox, score):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.score = score
        self.miss_count = 0


class SimpleTracker:
    """简单的在线跟踪器（用于评估）"""
    
    def __init__(self, max_miss=30, min_iou=0.5):
        self.max_miss = max_miss
        self.min_iou = min_iou
        self.tracks = {}
        self.next_id = 1
    
    def iou(self, box1, box2):
        """计算两个框的IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0
    
    def update(self, detections):
        """
        更新跟踪器
        Args:
            detections: list of [x, y, w, h, score]
        Returns:
            list of [x, y, w, h, track_id, score]
        """
        # 如果没有检测，增加所有跟踪的miss_count
        if len(detections) == 0:
            to_delete = []
            for tid, track in self.tracks.items():
                track.miss_count += 1
                if track.miss_count > self.max_miss:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return []
        
        # 匈牙利匹配简化版：贪心匹配
        matched_tracks = set()
        matched_dets = set()
        
        # 为每个跟踪找到最佳匹配
        for tid, track in self.tracks.items():
            best_iou = self.min_iou
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                iou = self.iou(track.bbox, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx >= 0:
                # 更新跟踪
                track.bbox = detections[best_det_idx][:4]
                track.score = detections[best_det_idx][4]
                track.miss_count = 0
                matched_tracks.add(tid)
                matched_dets.add(best_det_idx)
        
        # 未匹配的检测创建新跟踪
        for i, det in enumerate(detections):
            if i not in matched_dets:
                track = Track(self.next_id, det[:4], det[4])
                self.tracks[self.next_id] = track
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        # 未匹配的跟踪增加miss_count
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.miss_count += 1
                if track.miss_count > self.max_miss:
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self.tracks[tid]
        
        # 返回当前所有活跃的跟踪
        results = []
        for tid, track in self.tracks.items():
            x, y, w, h = track.bbox
            results.append([x, y, w, h, tid, track.score])
        
        return results


@torch.no_grad()
def evaluate_sequence(model, seq_path, device, conf_threshold=0.5, input_size=(800, 1333)):
    """评估单个序列"""
    model.eval()
    
    img_dir = seq_path / "img1"
    img_files = sorted(img_dir.glob("*.jpg"))
    
    transform = MOT17Transform(input_size=input_size, training=False)
    tracker = SimpleTracker()
    
    results = []
    
    for frame_id, img_file in enumerate(img_files, start=1):
        # 加载图像
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # 预处理
        result = transform(image)
        image_tensor = result["image"].unsqueeze(0).to(device)
        scale = result["scale"]
        
        # 推理
        outputs = model(image_tensor)
        
        # 解析输出
        logits = outputs['pred_logits'][0]  # [num_queries, num_classes+1]
        boxes = outputs['pred_boxes'][0]     # [num_queries, 4] (cxcywh normalized)
        
        # 获取置信度
        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)
        
        # 过滤背景和低于阈值的检测
        keep = (labels < logits.shape[-1] - 1) & (scores > conf_threshold)
        
        detections = []
        for i in range(len(keep)):
            if keep[i]:
                # 转换回xywh格式（图像坐标）
                cx, cy, w, h = boxes[i].cpu().numpy()
                cx *= input_size[1] / scale
                cy *= input_size[0] / scale
                w *= input_size[1] / scale
                h *= input_size[0] / scale
                
                x = cx - w / 2
                y = cy - h / 2
                
                score = scores[i].item()
                detections.append([x, y, w, h, score])
        
        # 更新跟踪器
        tracks = tracker.update(detections)
        
        # 记录结果
        for track in tracks:
            x, y, w, h, track_id, score = track
            results.append([
                frame_id, track_id, x, y, w, h, score, -1, -1, -1
            ])
    
    return results


def write_results(output_path, results):
    """写入MOT格式结果"""
    with open(output_path, 'w') as f:
        for row in results:
            f.write(','.join(map(str, map(int, row[:6]))) + ',' + 
                   ','.join(map(str, row[6:])) + '\n')


def main(args):
    """主函数"""
    print("=" * 60)
    print("MG-MOTRv2 MOT17 Evaluation")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载检查点
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 加载配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config.from_dict(config_dict)
        print("Loaded config from checkpoint")
    else:
        config = Config()
        print("Using default config")
    
    # 构建模型
    print("\nBuilding model...")
    model = build_model(config.to_dict()['model'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Loaded epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估序列
    data_root = Path(args.dataset_root)
    split_dir = data_root / args.split
    
    if not split_dir.exists():
        print(f"Error: Dataset not found at {split_dir}")
        return
    
    # 获取所有序列
    sequences = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(sequences)} sequences: {sequences}")
    
    # 评估每个序列
    for seq_name in sequences:
        print(f"\nEvaluating {seq_name}...")
        seq_path = split_dir / seq_name
        
        results = evaluate_sequence(
            model, seq_path, device,
            conf_threshold=args.conf_threshold,
            input_size=tuple(config.data['input_size'])
        )
        
        # 保存结果
        result_file = output_dir / f"{seq_name}.txt"
        write_results(result_file, results)
        print(f"  Saved {len(results)} detections to {result_file}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # 提示如何计算MOTA等指标
    print("\nTo compute MOTA, IDF1, etc.:")
    print("  1. Install motmetrics: pip install motmetrics")
    print("  2. Use official MOTChallenge evaluation tools")
    print("  3. Or use: python -m motmetrics.apps.eval_motchallenge \\")
    print(f"       {data_root}/{args.split} {output_dir}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
