"""
Training script for MG-MOTRv2
训练脚本 - MOT17数据集

使用方法：
    # 调试模式（快速验证）
    python -m mg_motrv2.train --debug
    
    # MOT17训练
    python -m mg_motrv2.train --dataset_root ./data/MOT17 --batch_size 2 --epochs 300
    
    # 使用所有检测器（数据增强）
    python -m mg_motrv2.train --dataset_root ./data/MOT17 --use_all_detectors
    
    # 恢复训练
    python -m mg_motrv2.train --resume ./outputs/checkpoint_0100.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.mg_motrv2 import build_model
from .configs.default_config import Config, get_debug_config
from .utils import build_matcher, build_criterion
from .datasets import build_mot17_dataloader


def get_args_parser():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser('MG-MOTRv2 Training on MOT17')
    
    # 配置文件
    parser.add_argument('--config', default='', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='输出目录')
    
    # 数据集配置
    parser.add_argument('--dataset_root', default='./data/MOT17', type=str, help='MOT17数据集根目录')
    parser.add_argument('--use_all_detectors', action='store_true', help='使用所有检测器版本（数据增强）')
    
    # 模型配置
    parser.add_argument('--backbone', default='resnet50', type=str, help='骨干网络类型')
    parser.add_argument('--d_model', default=256, type=int, help='特征维度')
    parser.add_argument('--num_queries', default=300, type=int, help='查询数量')
    
    # 训练配置
    parser.add_argument('--lr', default=2e-4, type=float, help='学习率')
    parser.add_argument('--lr_backbone', default=2e-5, type=float, help='骨干网络学习率')
    parser.add_argument('--batch_size', default=2, type=int, help='批次大小')
    parser.add_argument('--epochs', default=300, type=int, help='训练轮数')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
    parser.add_argument('--lr_drop', default=200, type=int, help='学习率衰减轮数')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='梯度裁剪阈值')
    parser.add_argument('--num_workers', default=4, type=int, help='数据加载 workers')
    
    # 输入尺寸
    parser.add_argument('--input_height', default=800, type=int, help='输入图像高度')
    parser.add_argument('--input_width', default=1333, type=int, help='输入图像宽度')
    
    # 其他
    parser.add_argument('--device', default='cuda', type=str, help='设备')
    parser.add_argument('--resume', default='', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--eval_interval', default=5, type=int, help='评估间隔（轮数）')
    parser.add_argument('--save_interval', default=10, type=int, help='保存间隔（轮数）')
    parser.add_argument('--debug', action='store_true', help='调试模式（快速验证）')
    parser.add_argument('--print_freq', default=50, type=int, help='打印频率（迭代次数）')
    
    return parser


def build_optimizer(model: nn.Module, config: Config):
    """构建优化器（骨干网络使用较小学习率）"""
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': other_params, 'lr': config.train['lr']},
        {'params': backbone_params, 'lr': config.train['lr_backbone']}
    ]
    
    return torch.optim.AdamW(param_groups, weight_decay=config.train['weight_decay'])


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    clip_max_norm: float,
    print_freq: int,
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    criterion.train()
    
    total_loss = 0.0
    loss_ce_sum = 0.0
    loss_bbox_sum = 0.0
    loss_giou_sum = 0.0
    
    num_batches = len(dataloader)
    
    for i, (images, targets) in enumerate(dataloader):
        # 将数据移到设备
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        
        # 加权损失
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        
        # 梯度裁剪
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        
        optimizer.step()
        
        # 统计损失
        total_loss += losses.item()
        loss_ce_sum += loss_dict.get('loss_ce', 0).item() if isinstance(loss_dict.get('loss_ce'), torch.Tensor) else 0
        loss_bbox_sum += loss_dict.get('loss_bbox', 0).item() if isinstance(loss_dict.get('loss_bbox'), torch.Tensor) else 0
        loss_giou_sum += loss_dict.get('loss_giou', 0).item() if isinstance(loss_dict.get('loss_giou'), torch.Tensor) else 0
        
        # 打印进度
        if (i + 1) % print_freq == 0 or i == num_batches - 1:
            avg_loss = total_loss / (i + 1)
            print(
                f"Epoch [{epoch}] [{i+1}/{num_batches}] "
                f"Loss: {avg_loss:.4f} "
                f"(ce: {loss_ce_sum/(i+1):.4f}, "
                f"bbox: {loss_bbox_sum/(i+1):.4f}, "
                f"giou: {loss_giou_sum/(i+1):.4f})"
            )
    
    return {
        'loss': total_loss / num_batches,
        'loss_ce': loss_ce_sum / num_batches,
        'loss_bbox': loss_bbox_sum / num_batches,
        'loss_giou': loss_giou_sum / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    criterion.eval()
    
    total_loss = 0.0
    loss_ce_sum = 0.0
    loss_bbox_sum = 0.0
    loss_giou_sum = 0.0
    
    num_batches = len(dataloader)
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        total_loss += losses.item()
        loss_ce_sum += loss_dict.get('loss_ce', 0).item() if isinstance(loss_dict.get('loss_ce'), torch.Tensor) else 0
        loss_bbox_sum += loss_dict.get('loss_bbox', 0).item() if isinstance(loss_dict.get('loss_bbox'), torch.Tensor) else 0
        loss_giou_sum += loss_dict.get('loss_giou', 0).item() if isinstance(loss_dict.get('loss_giou'), torch.Tensor) else 0
    
    return {
        'loss': total_loss / num_batches,
        'loss_ce': loss_ce_sum / num_batches,
        'loss_bbox': loss_bbox_sum / num_batches,
        'loss_giou': loss_giou_sum / num_batches,
    }


def main(args):
    """主函数"""
    print("=" * 60)
    print("MG-MOTRv2 Training on MOT17")
    print("=" * 60)
    
    # 加载配置
    if args.debug:
        print("Using debug configuration")
        config = get_debug_config()
    else:
        print("Using standard configuration")
        config = Config()
        config.model['backbone']['type'] = args.backbone
        config.model['d_model'] = args.d_model
        config.model['num_queries'] = args.num_queries
        config.train['lr'] = args.lr
        config.train['lr_backbone'] = args.lr_backbone
        config.train['batch_size'] = args.batch_size
        config.train['epochs'] = args.epochs
        config.train['weight_decay'] = args.weight_decay
        config.train['lr_drop'] = args.lr_drop
        config.train['clip_max_norm'] = args.clip_max_norm
        config.data['num_workers'] = args.num_workers
        config.data['input_size'] = [args.input_height, args.input_width]
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # 构建数据加载器
    print("\nBuilding data loaders...")
    print(f"Dataset root: {args.dataset_root}")
    
    train_loader = build_mot17_dataloader(
        data_root=args.dataset_root,
        batch_size=config.train['batch_size'],
        split="train",
        num_workers=config.data['num_workers'],
        input_size=tuple(config.data['input_size']),
        use_all_detectors=args.use_all_detectors,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    
    # 构建模型
    print("\nBuilding model...")
    model = build_model(config.to_dict()['model'])
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params:,}")
    
    # 构建匹配器和损失函数
    matcher = build_matcher(config.to_dict()['train'])
    criterion = build_criterion(config.to_dict(), matcher)
    criterion.to(device)
    
    # 构建优化器
    optimizer = build_optimizer(model, config)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.train['lr_drop']
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("Start training")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config.train['epochs']):
        epoch_start_time = time.time()
        
        # 训练
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch, config.train['clip_max_norm'], args.print_freq
        )
        
        # 学习率调整
        lr_scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        print(
            f"Epoch [{epoch}] completed in {epoch_time:.1f}s | "
            f"Loss: {train_losses['loss']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # 保存检查点
        is_best = train_losses['loss'] < best_loss
        if is_best:
            best_loss = train_losses['loss']
        
        if (epoch + 1) % args.save_interval == 0 or is_best or epoch == config.train['epochs'] - 1:
            checkpoint_path = output_dir / f'checkpoint_{epoch:04d}.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config': config.to_dict(),
                'train_losses': train_losses,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            if is_best:
                best_path = output_dir / 'checkpoint_best.pth'
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'train_losses': train_losses,
                }, best_path)
                print(f"Saved best checkpoint to {best_path}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
