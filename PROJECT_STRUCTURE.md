# MG-MOTRv2 项目结构（精简版）

## 概述

基于 MOTRv2 架构的多粒度注意力机制多目标跟踪研究框架 - **精简版**

保留核心功能，去除不必要的复杂部分，预留扩展空间。

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
│         └─────────────┘                 │
└─────────────────────────────────────────┘
```

**组件**：
- `GranularityLevel`: 单层粒度注意力
- `MultiGranularityAttention`: 多粒度融合（adaptive/concat/sum）
- `TemporalGranularityAttention`: 时序多粒度（可选扩展）

### 2. 骨干网络 (`models/backbone.py`)

**架构**：
```
Input Image
    ↓
ResNet (Backbone)
    ↓
FPN (Feature Pyramid)
    ↓
Multi-Granularity Features:
  - Fine   [B, C, H/4, W/4]
  - Medium [B, C, H/8, W/8]
  - Coarse [B, C, H/16, W/16]
```

### 3. MG-DETR头 (`models/mg_motr.py`)

**组件**：
- `MGTransformerEncoder`: 编码器（后半部分使用MG-Attention）
- `MGTransformerDecoder`: 标准解码器
- `MGTracker`: 轨迹管理（简化版贪心匹配）

### 4. 完整模型 (`models/mg_motrv2.py`)

**数据流**：
```
Input Image
    ↓
MultiGranularityBackbone
    ↓
{Fine, Medium, Coarse} Features
    ↓
MG_DETRHead (Encoder + Decoder)
    ↓
Detection Outputs
    ↓
MGTracker
    ↓
Tracking Results
```

## 配置文件 (`configs/default_config.py`)

**配置项**：
- `model`: 模型架构参数
- `train`: 训练参数
- `data`: 数据配置
- `log`: 日志配置

**预定义配置**：
- `get_base_config()`: 基础配置
- `get_debug_config()`: 调试配置（快速验证）

## 工具函数

### 匹配器 (`utils/matcher.py`)
- `HungarianMatcher`: 匈牙利匹配器
- 支持分类/L1/GIoU代价

### 损失函数 (`utils/losses.py`)
- `SetCriterion`: DETR风格集合损失
  - 分类损失（交叉熵）
  - 边界框损失（L1 + GIoU）

## 脚本

### 演示 (`demo.py`)
```bash
python -m mg_motrv2.demo --all       # 运行所有演示
python -m mg_motrv2.demo --mg-attn   # 多粒度注意力
python -m mg_motrv2.demo --backbone  # 骨干网络
python -m mg_motrv2.demo --model     # 完整模型
```

### 训练 (`train.py`)
```bash
python -m mg_motrv2.train --debug    # 快速验证
python -m mg_motrv2.train --dataset mot17 --batch_size 2
```

### 测试 (`test_basic.py`)
```bash
python test_basic.py  # 验证所有核心模块
```

## 扩展接口

### 1. 添加新粒度
```python
# mg_attention.py 中修改 stride 字典
self.stride = {
    "ultra_fine": 0.5,  # 新粒度
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

### 3. 添加新损失
```python
# losses.py 中扩展
self.losses.append("custom")
def loss_custom(self, outputs, targets, indices, num_boxes):
    return {"loss_custom": self.custom_loss(outputs, targets)}
```

## 预留改进空间

1. **骨干网络预训练权重加载**
2. **数据加载器实现**
3. **ReID特征学习**
4. **Deformable Attention**（性能优化）
5. **更精细的时序对齐**
6. **评估指标计算**（MOTA, IDF1等）
7. **TensorBoard/Wandb集成**

## 文件清单

```
mg_motrv2/
├── __init__.py              # 包初始化
├── models/
│   ├── __init__.py          # 模型导出
│   ├── mg_attention.py      # 多粒度注意力 (~7KB)
│   ├── backbone.py          # 骨干网络 (~7KB)
│   ├── mg_motr.py          # MG-DETR头 (~13KB)
│   └── mg_motrv2.py        # 完整模型 (~5KB)
├── configs/
│   └── default_config.py   # 默认配置 (~3KB)
├── utils/
│   ├── __init__.py          # 工具导出
│   ├── matcher.py          # 匈牙利匹配器 (~4KB)
│   └── losses.py           # 损失函数 (~5KB)
├── train.py                # 训练脚本 (~5KB)
└── demo.py                 # 演示脚本 (~6KB)

根目录：
├── README.md               # 项目文档 (~2KB)
├── PROJECT_STRUCTURE.md    # 本文件
├── requirements.txt        # 依赖项
├── setup.py               # 安装脚本
└── test_basic.py          # 基础测试 (~6KB)
```

## 代码统计

- **总代码行数**：~1800行（精简后）
- **Python文件数**：12个
- **核心模块数**：4个

## 运行要求

- Python >= 3.8
- PyTorch >= 1.9.0
- 显存：至少 8GB（训练）
