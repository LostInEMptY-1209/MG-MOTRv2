# MG-MOTRv2 项目结构

## 概述

本项目基于 MOTRv2 架构，实现了**多粒度注意力机制（Multi-Granularity Attention）**的多目标跟踪研究框架。

## 核心模块

### 1. 多粒度注意力机制 (`models/mg_attention.py`)

```
┌─────────────────────────────────────────┐
│     Multi-Granularity Attention         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  Fine   │ │ Medium  │ │ Coarse  │   │
│  │(高分辨率)│ │(中分辨率)│ │(低分辨率)│   │
│  └────┬────┘ └────┬────┘ └────┬────┘   │
│       └───────────┼───────────┘         │
│                   ↓                     │
│         ┌─────────────┐                 │
│         │   Fusion    │  ← 自适应融合   │
│         │ (adaptive/  │    concat/sum  │
│         │ concat/sum) │                 │
│         └─────────────┘                 │
└─────────────────────────────────────────┘
```

**主要组件：**
- `GranularityLevel`: 单一层级的粒度注意力模块
- `MultiGranularityAttention`: 整合3个粒度层级的注意力模块
- `TemporalGranularityAttention`: 时序多粒度注意力（支持视频序列）

### 2. MG-DETR 头 (`models/mg_motr.py`)

**主要组件：**
- `MG_DETRHead`: 多粒度DETR检测头
- `MGTransformerEncoder`: 多粒度Transformer编码器
- `MGTransformerDecoder`: Transformer解码器
- `MGTracker`: 轨迹管理器（生命周期管理）

### 3. 骨干网络 (`models/backbone.py`)

**架构：**
```
Input Image
    ↓
ResNet (backbone)
    ↓
FPN (Feature Pyramid Network)
    ↓
Multi-Granularity Features:
  - Fine   [B, C, H/4, W/4]
  - Medium [B, C, H/8, W/8]
  - Coarse [B, C, H/16, W/16]
```

### 4. 完整模型集成 (`models/mg_motrv2.py`)

**完整数据流：**
```
Input Image [B, 3, H, W]
    ↓
MultiGranularityBackbone
    ↓
Multi-Granularity Features {fine, medium, coarse}
    ↓
MG_DETRHead (Encoder + Decoder with MG-Attention)
    ↓
Detection Outputs + Track Queries
    ↓
MGTracker
    ↓
Tracking Results {id, bbox, score, ...}
```

## 配置文件 (`configs/default_config.py`)

**配置类别：**
- `model`: 模型架构参数
- `train`: 训练参数（学习率、批次大小等）
- `data`: 数据配置（数据集、增强等）
- `test`: 测试参数
- `log`: 日志配置
- `device`: 硬件配置

**预定义配置：**
- `get_base_config()`: 基础配置
- `get_mot17_config()`: MOT17数据集配置
- `get_dance_config()`: DanceTrack数据集配置（启用时序）
- `get_debug_config()`: 调试配置（快速验证）

## 工具函数

### 匹配器 (`utils/matcher.py`)
- `HungarianMatcher`: 匈牙利匹配器（预测与真值的最优匹配）
- 支持分类代价、边界框L1代价、GIoU代价

### 损失函数 (`utils/losses.py`)
- `SetCriterion`: DETR风格的集合损失
  - 分类损失（交叉熵）
  - 边界框损失（L1 + GIoU）
- `TrackingLoss`: 跟踪专用损失
  - ReID对比损失
  - 时序一致性损失

## 脚本

### 训练脚本 (`train.py`)
```bash
# 基础训练
python -m mg_motrv2.train --dataset mot17 --batch_size 2

# 调试模式
python -m mg_motrv2.train --debug
```

### 演示脚本 (`demo.py`)
```bash
# 运行所有演示
python -m mg_motrv2.demo --all

# 特定演示
python -m mg_motrv2.demo --granularity
python -m mg_motrv2.demo --mg-attn
python -m mg_motrv2.demo --temporal
python -m mg_motrv2.demo --model
python -m mg_motrv2.demo --structure
```

## 扩展接口

### 1. 添加新粒度级别
```python
# 在 GranularityLevel 中添加
self.stride = {
    "ultra_fine": 0.5,   # 新粒度
    "fine": 1,
    "medium": 2,
    "coarse": 4,
}[granularity]
```

### 2. 自定义融合策略
```python
class CustomFusionMGAttention(MultiGranularityAttention):
    def forward(self, features):
        # 自定义融合逻辑
        return fused
```

### 3. 自定义损失
```python
class CustomLoss(nn.Module):
    def forward(self, predictions, targets):
        return loss

# 在 SetCriterion 中注册
self.losses.append("custom")
```

## 预留改进空间

当前实现预留了以下扩展点：

1. **骨干网络预训练权重加载** (ResNetBackbone)
2. **数据加载器实现** (data/)
3. **ReID特征学习** (TrackingLoss)
4. **Deformable Attention** (性能优化)
5. **更精细的时序对齐** (TemporalFeatureFusion)
6. **评估指标计算** (MOTA, IDF1等)
7. **TensorBoard/Wandb集成**
8. **分布式训练支持**

## 文件清单

```
mg_motrv2/
├── __init__.py              # 包初始化
├── models/
│   ├── __init__.py          # 模型导出
│   ├── mg_attention.py      # 多粒度注意力（9080字节）
│   ├── mg_motr.py          # MG-DETR头（15479字节）
│   ├── backbone.py         # 骨干网络（11770字节）
│   └── mg_motrv2.py        # 完整模型（9372字节）
├── configs/
│   └── default_config.py   # 默认配置（6075字节）
├── utils/
│   ├── __init__.py          # 工具导出
│   ├── matcher.py          # 匈牙利匹配器（5253字节）
│   └── losses.py           # 损失函数（8573字节）
├── train.py                # 训练脚本（8642字节）
└── demo.py                 # 演示脚本（7921字节）

根目录：
├── README.md               # 项目文档（5706字节）
├── PROJECT_STRUCTURE.md    # 本文件
├── requirements.txt        # 依赖项（379字节）
├── setup.py               # 安装脚本（1299字节）
├── test_basic.py          # 基础测试（5872字节）
└── .gitignore             # Git忽略配置（474字节）
```

## 代码统计

- 总代码行数：~2500+ 行
- Python文件数：13个
- 核心模块数：4个
- 配置选项数：50+

## 运行要求

- Python >= 3.8
- PyTorch >= 1.9.0
- 显存：至少 8GB（训练模式）
- 内存：至少 16GB
