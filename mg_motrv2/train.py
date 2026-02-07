"""
Training script for MG-MOTRv2
训练脚本 - 简化版

使用方法：
    python -m mg_motrv2.train --debug  # 快速验证
    python -m mg_motrv2.train --dataset mot17 --batch_size 2
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from models.mg_motrv2 import build_model
from configs.default_config import Config, get_debug_config
from utils import build_matcher, build_criterion


def get_args_parser():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser('MG-MOTRv2 Training')
    
    parser.add_argument('--config', default='', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', default='./outputs', type=str)
    
    # 模型配置
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    
    # 训练配置
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    
    # 其他
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
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
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
):
    """训练一个epoch（简化版）"""
    model.train()
    
    # TODO: 实现数据加载和训练循环
    # 这里仅作为示例框架
    print(f"Epoch [{epoch}] - Training (placeholder)")
    
    return {'total': 0.0}


def main(args):
    """主函数"""
    # 加载配置
    if args.debug:
        config = get_debug_config()
    else:
        config = Config()
        config.model['backbone']['type'] = args.backbone
        config.model['d_model'] = args.d_model
        config.model['num_queries'] = args.num_queries
        config.train['lr'] = args.lr
        config.train['batch_size'] = args.batch_size
        config.train['epochs'] = args.epochs
    
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.train['lr_drop']
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    
    # 训练循环
    print("Start training")
    for epoch in range(start_epoch, config.train['epochs']):
        train_losses = train_one_epoch(model, criterion, optimizer, device, epoch)
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
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
