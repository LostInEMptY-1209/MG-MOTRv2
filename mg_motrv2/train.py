"""
Training script for MG-MOTRv2
MG-MOTRv2训练脚本
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mg_motrv2 import build_model
from configs.default_config import Config, get_base_config, get_debug_config
from utils import build_matcher, build_criterion


def get_args_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser('MG-MOTRv2 Training', add_help=False)
    
    # 基础配置
    parser.add_argument('--config', default='', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='输出目录')
    
    # 模型配置
    parser.add_argument('--backbone', default='resnet50', type=str, help='骨干网络')
    parser.add_argument('--d_model', default=256, type=int, help='模型维度')
    parser.add_argument('--num_queries', default=300, type=int, help='查询数量')
    
    # 训练配置
    parser.add_argument('--lr', default=2e-4, type=float, help='学习率')
    parser.add_argument('--lr_backbone', default=2e-5, type=float, help='骨干网络学习率')
    parser.add_argument('--batch_size', default=2, type=int, help='批次大小')
    parser.add_argument('--epochs', default=300, type=int, help='训练轮数')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
    
    # 数据配置
    parser.add_argument('--dataset', default='mot17', type=str, help='数据集')
    parser.add_argument('--data_root', default='./data', type=str, help='数据根目录')
    
    # 其他
    parser.add_argument('--device', default='cuda', type=str, help='训练设备')
    parser.add_argument('--num_workers', default=4, type=int, help='数据加载线程数')
    parser.add_argument('--resume', default='', type=str, help='恢复训练的检查点')
    parser.add_argument('--eval', action='store_true', help='仅评估模式')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    return parser


def build_optimizer(model: nn.Module, config: Config):
    """
    构建优化器
    
    对骨干网络和主体部分使用不同的学习率
    """
    # 分离骨干网络参数和主体参数
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
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.train['weight_decay']
    )
    
    return optimizer


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: Config):
    """构建学习率调度器"""
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train['lr_drop']
    )
    return lr_scheduler


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Config
):
    """训练一个epoch"""
    model.train()
    criterion.train()
    
    total_loss = 0
    loss_dict_sum = {}
    
    for i, (samples, targets) in enumerate(data_loader):
        # 数据移动到设备
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        outputs = model(samples, targets)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(
            loss_dict[k] * weight_dict[k] 
            for k in loss_dict.keys() if k in weight_dict
        )
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        
        # 梯度裁剪
        if config.train['clip_max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.train['clip_max_norm']
            )
        
        optimizer.step()
        
        # 记录
        total_loss += losses.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_sum:
                loss_dict_sum[k] = 0
            loss_dict_sum[k] += v.item()
        
        # 打印进度
        if (i + 1) % config.log['log_interval'] == 0:
            avg_loss = total_loss / (i + 1)
            print(f"Epoch [{epoch}] [{i+1}/{len(data_loader)}] Loss: {avg_loss:.4f}")
    
    # 返回平均损失
    avg_losses = {k: v / len(data_loader) for k, v in loss_dict_sum.items()}
    avg_losses['total'] = total_loss / len(data_loader)
    
    return avg_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device
):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    total_loss = 0
    loss_dict_sum = {}
    
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] 
            for k in loss_dict.keys() if k in weight_dict
        )
        
        total_loss += losses.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_sum:
                loss_dict_sum[k] = 0
            loss_dict_sum[k] += v.item()
    
    avg_losses = {k: v / len(data_loader) for k, v in loss_dict_sum.items()}
    avg_losses['total'] = total_loss / len(data_loader)
    
    return avg_losses


def main(args):
    """主函数"""
    # 加载配置
    if args.debug:
        config = get_debug_config()
    else:
        config = get_base_config()
        
        # 更新配置
        config.model['backbone']['type'] = args.backbone
        config.model['d_model'] = args.d_model
        config.model['num_queries'] = args.num_queries
        config.train['lr'] = args.lr
        config.train['lr_backbone'] = args.lr_backbone
        config.train['batch_size'] = args.batch_size
        config.train['epochs'] = args.epochs
        config.data['dataset'] = args.dataset
        config.data['data_root'] = args.data_root
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # 构建模型
    print("Building model...")
    model = build_model(config.to_dict()['model'])
    model.to(device)
    
    # 构建匹配器和损失函数
    matcher = build_matcher(config.to_dict()['train'])
    criterion = build_criterion(config.to_dict(), matcher)
    criterion.to(device)
    
    # 构建优化器
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    
    # 评估模式
    if args.eval:
        print("Evaluation mode")
        # 这里需要加载验证数据
        # val_losses = evaluate(model, criterion, val_loader, device)
        # print(f"Validation losses: {val_losses}")
        return
    
    # 训练循环
    print("Start training")
    for epoch in range(start_epoch, config.train['epochs']):
        # 训练
        # train_losses = train_one_epoch(
        #     model, criterion, train_loader, optimizer, 
        #     device, epoch, config
        # )
        
        lr_scheduler.step()
        
        # 保存检查点
        if (epoch + 1) % config.log['checkpoint_interval'] == 0:
            checkpoint_path = output_dir / f'checkpoint_{epoch:04d}.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config': config.to_dict()
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MG-MOTRv2 Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
