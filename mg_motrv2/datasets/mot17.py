"""
MOT17 Dataset Loader
MOT17多目标跟踪数据集加载器

数据格式:
    - 图像: MOT17/train/<seq_name>/img1/000001.jpg
    - 标注: MOT17/train/<seq_name>/gt/gt.txt
      格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MOT17Transform:
    """MOT17数据预处理"""
    
    def __init__(self, input_size: Tuple[int, int] = (800, 1333), training: bool = True):
        self.input_size = input_size  # [H, W]
        self.training = training
        
        # 标准化参数 (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __call__(self, image: np.ndarray, targets: Optional[Dict] = None):
        """
        Args:
            image: [H, W, 3] numpy array (BGR)
            targets: dict with 'boxes' [N, 4] in xywh format, 'labels' [N], 'track_ids' [N]
        """
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]
        
        # 计算缩放比例
        target_h, target_w = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # 调整图像大小
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充后的图像
        padded_image = np.zeros((target_h, target_w, 3), dtype=np.float32)
        padded_image[:new_h, :new_w] = image
        
        # 转换为tensor并归一化到[0, 1]
        image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
        
        # ImageNet标准化
        image_tensor = (image_tensor - self.mean) / self.std
        
        result = {"image": image_tensor, "scale": scale}
        
        if targets is not None:
            boxes = targets["boxes"].copy()
            
            # 调整框的坐标 (xywh -> cxcywh 归一化)
            boxes[:, [0, 2]] *= scale  # x, w
            boxes[:, [1, 3]] *= scale  # y, h
            
            # 转换为center_x, center_y, w, h 并归一化
            boxes_xyxy = boxes.copy()
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
            
            # 转换为cxcywh格式并归一化到[0, 1]
            boxes_cxcywh = np.zeros_like(boxes)
            boxes_cxcywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / target_w  # cx
            boxes_cxcywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / target_h  # cy
            boxes_cxcywh[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / target_w      # w
            boxes_cxcywh[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / target_h      # h
            
            result["targets"] = {
                "boxes": torch.from_numpy(boxes_cxcywh).float(),
                "labels": torch.zeros(len(boxes), dtype=torch.long),  # 所有目标都是前景类0
                "track_ids": torch.from_numpy(targets["track_ids"]).long(),
                "orig_size": torch.tensor([orig_h, orig_w]),
            }
        
        return result


class MOT17Dataset(Dataset):
    """MOT17数据集"""
    
    # MOT17训练序列
    TRAIN_SEQUENCES = [
        "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"
    ]
    
    # 检测器后缀
    DETECTORS = ["DPM", "FRCNN", "SDP"]
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[MOT17Transform] = None,
        use_all_detectors: bool = False,
        min_visibility: float = 0.0,
        min_bbox_area: float = 100.0,
    ):
        """
        Args:
            data_root: MOT17数据集根目录
            split: 'train' 或 'test'
            transform: 数据预处理
            use_all_detectors: 是否使用所有检测器的结果（每个序列有3个版本）
            min_visibility: 最小可见度阈值
            min_bbox_area: 最小边界框面积
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform or MOT17Transform()
        self.use_all_detectors = use_all_detectors
        self.min_visibility = min_visibility
        self.min_bbox_area = min_bbox_area
        
        # 收集所有样本
        self.samples = self._collect_samples()
        print(f"MOT17 {split}: {len(self.samples)} frames loaded")
    
    def _collect_samples(self) -> List[Dict]:
        """收集所有样本"""
        samples = []
        
        sequences = self.TRAIN_SEQUENCES if self.split == "train" else []
        
        for seq_name in sequences:
            if self.use_all_detectors:
                # 使用所有检测器版本
                for detector in self.DETECTORS:
                    seq_full_name = f"{seq_name}-{detector}"
                    samples.extend(self._collect_sequence_samples(seq_full_name))
            else:
                # 只使用FRCNN版本
                seq_full_name = f"{seq_name}-FRCNN"
                samples.extend(self._collect_sequence_samples(seq_full_name))
        
        return samples
    
    def _collect_sequence_samples(self, seq_name: str) -> List[Dict]:
        """收集单个序列的所有样本"""
        samples = []
        seq_path = self.data_root / self.split / seq_name
        
        if not seq_path.exists():
            print(f"Warning: Sequence {seq_name} not found at {seq_path}")
            return samples
        
        img_dir = seq_path / "img1"
        gt_file = seq_path / "gt" / "gt.txt"
        
        if not gt_file.exists():
            print(f"Warning: GT file not found: {gt_file}")
            return samples
        
        # 读取标注文件
        annotations = self._load_annotations(gt_file)
        
        # 获取所有图像文件
        img_files = sorted(img_dir.glob("*.jpg"))
        
        for img_file in img_files:
            frame_id = int(img_file.stem)
            if frame_id in annotations:
                samples.append({
                    "image_path": str(img_file),
                    "seq_name": seq_name,
                    "frame_id": frame_id,
                    "annotations": annotations[frame_id],
                })
        
        return samples
    
    def _load_annotations(self, gt_file: Path) -> Dict[int, List[Dict]]:
        """加载标注文件"""
        annotations = {}
        
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                bbox_left = float(parts[2])
                bbox_top = float(parts[3])
                bbox_width = float(parts[4])
                bbox_height = float(parts[5])
                conf = float(parts[6])
                
                # 类别和可见度（如果有）
                class_id = int(parts[7]) if len(parts) > 7 else 1
                visibility = float(parts[8]) if len(parts) > 8 else 1.0
                
                # 过滤条件：
                # 1. 只保留行人 (class_id == 1)
                # 2. 置信度 > 0
                # 3. 可见度 >= 阈值
                # 4. 边界框面积 >= 阈值
                if class_id != 1 or conf <= 0:
                    continue
                
                if visibility < self.min_visibility:
                    continue
                
                bbox_area = bbox_width * bbox_height
                if bbox_area < self.min_bbox_area:
                    continue
                
                if frame_id not in annotations:
                    annotations[frame_id] = []
                
                annotations[frame_id].append({
                    "track_id": track_id,
                    "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                    "visibility": visibility,
                })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: [3, H, W] tensor
            target: dict with 'boxes', 'labels', 'track_ids'
        """
        sample = self.samples[idx]
        
        # 加载图像
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        
        # 准备目标
        anns = sample["annotations"]
        if len(anns) > 0:
            boxes = np.array([ann["bbox"] for ann in anns])
            track_ids = np.array([ann["track_id"] for ann in anns])
        else:
            boxes = np.zeros((0, 4))
            track_ids = np.array([])
        
        targets = {
            "boxes": boxes,
            "track_ids": track_ids,
        }
        
        # 应用变换
        result = self.transform(image, targets)
        
        return result["image"], result["targets"]


def build_mot17_dataloader(
    data_root: str,
    batch_size: int = 2,
    split: str = "train",
    num_workers: int = 4,
    input_size: Tuple[int, int] = (800, 1333),
    use_all_detectors: bool = False,
) -> DataLoader:
    """构建MOT17数据加载器"""
    
    transform = MOT17Transform(input_size=input_size, training=(split == "train"))
    
    dataset = MOT17Dataset(
        data_root=data_root,
        split=split,
        transform=transform,
        use_all_detectors=use_all_detectors,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=mot17_collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    
    return dataloader


def mot17_collate_fn(batch):
    """自定义collate函数处理变长目标"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets
