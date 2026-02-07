# MG-MOTRv2: Multi-Granularity Multi-Object Tracking with Transformer

多粒度多目标跟踪Transformer研究框架

## 项目简介

本项目基于 MOTRv2 架构，引入**多粒度注意力机制（Multi-Granularity Attention）**，用于提升多目标跟踪（MOT）任务的性能。

### 核心创新

- **多粒度注意力机制**：整合细粒度（fine）、中粒度（medium）、粗粒度（coarse）三个空间分辨率层级的特征
- **自适应融合**：根据输入动态调整不同粒度的权重
- **时序建模**：支持视频序列的时序多粒度注意力
- **模块化设计**：便于扩展和实验

## 项目结构

```
mg_motrv2/
├── models/                    # 核心模型模块
│   ├── __init__.py
│   ├── mg_attention.py        # 多粒度注意力核心实现
│   ├── mg_motr.py            # MG-DETR头和跟踪器
│   ├── backbone.py           # 多粒度骨干网络
│   └── mg_motrv2.py          # 完整模型集成
├── configs/                   # 配置文件
│   └── default_config.py     # 默认配置
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── matcher.py            # 匈牙利匹配器
│   └── losses.py             # 损失函数
├── train.py                   # 训练脚本
├── demo.py                    # 演示脚本
└── __init__.py
```

## 核心模块

### 1. 多粒度注意力 (Multi-Granularity Attention)

```python
from models import MultiGranularityAttention

# 创建多粒度注意力模块
mg_attn = MultiGranularityAttention(
    d_model=256,           # 特征维度
    n_heads=8,             # 注意力头数
    n_levels=3,            # 粒度级别数（fine, medium, coarse）
    fusion_type="adaptive" # 融合类型: adaptive/concat/sum
)

# 输入多粒度特征
features = [fine_feat, medium_feat, coarse_feat]
fused_output = mg_attn(features)
```

### 2. 时序多粒度注意力 (Temporal Granularity Attention)

```python
from models import TemporalGranularityAttention

# 创建时序模块
temporal_mg_attn = TemporalGranularityAttention(
    d_model=256,
    n_heads=8,
    n_frames=4,    # 处理的帧数
    n_levels=3
)

# 输入多帧多粒度特征
frame_features = [[fine_t, medium_t, coarse_t] for t in range(n_frames)]
output = temporal_mg_attn(frame_features)
```

### 3. 多粒度骨干网络

```python
from models import MultiGranularityBackbone

# 创建骨干网络
backbone = MultiGranularityBackbone(
    backbone_type="resnet50",
    fpn_channels=256,
    granularity_levels=["fine", "medium", "coarse"]
)

# 提取多粒度特征
mg_features = backbone(images)  # dict: {fine: tensor, medium: tensor, coarse: tensor}
```

### 4. 完整模型

```python
from models.mg_motrv2 import MGMOTRv2, build_model
from configs.default_config import Config

# 使用配置创建模型
config = Config()
model = build_model(config.to_dict()['model'])

# 前向传播
outputs = model(images)
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行演示

```bash
# 运行所有演示
python -m mg_motrv2.demo --all

# 运行特定演示
python -m mg_motrv2.demo --granularity    # 粒度层演示
python -m mg_motrv2.demo --mg-attn        # 多粒度注意力演示
python -m mg_motrv2.demo --temporal       # 时序注意力演示
python -m mg_motrv2.demo --backbone       # 骨干网络演示
python -m mg_motrv2.demo --model          # 完整模型演示
python -m mg_motrv2.demo --structure      # 模型结构说明
```

### 训练模型

```bash
# 基础训练
python -m mg_motrv2.train \
    --dataset mot17 \
    --data_root ./data \
    --batch_size 2 \
    --epochs 200

# 使用特定骨干网络
python -m mg_motrv2.train \
    --backbone resnet101 \
    --d_model 256 \
    --lr 2e-4

# 恢复训练
python -m mg_motrv2.train \
    --resume ./outputs/checkpoint_0100.pth

# 调试模式（快速验证）
python -m mg_motrv2.train --debug
```

## 配置说明

### 模型配置 (`model`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `d_model` | 特征维度 | 256 |
| `num_queries` | 查询数量 | 300 |
| `num_classes` | 类别数 | 1 |
| `granularity_levels` | 粒度级别 | ["fine", "medium", "coarse"] |
| `use_temporal` | 使用时序建模 | False |
| `n_frames` | 时序帧数 | 1 |

### 训练配置 (`train`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `batch_size` | 批次大小 | 2 |
| `lr` | 学习率 | 2e-4 |
| `lr_backbone` | 骨干网络学习率 | 2e-5 |
| `epochs` | 训练轮数 | 300 |
| `lr_drop` | 学习率衰减轮数 | 200 |
| `clip_max_norm` | 梯度裁剪阈值 | 0.1 |

### 多粒度注意力配置 (`mg_attention`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `n_heads` | 注意力头数 | 8 |
| `fusion_type` | 融合类型 | "adaptive" |
| `dropout` | Dropout率 | 0.1 |

融合类型说明：
- `adaptive`: 自适应权重融合（根据特征动态调整）
- `concat`: 拼接后投影
- `sum`: 直接相加

## 扩展接口

本项目预留了以下扩展接口：

### 1. 添加新的粒度级别

```python
# 在 mg_attention.py 中扩展 GranularityLevel
class GranularityLevel(nn.Module):
    def __init__(self, ..., granularity="ultra_fine"):
        self.stride = {
            "ultra_fine": 0.5,  # 上采样
            "fine": 1,
            "medium": 2,
            "coarse": 4,
            "ultra_coarse": 8   # 新粒度
        }[granularity]
```

### 2. 自定义融合策略

```python
# 继承 MultiGranularityAttention 并重写融合逻辑
class CustomFusionMGAttention(MultiGranularityAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 自定义融合模块
        self.custom_fusion = nn.Sequential(...)
    
    def forward(self, features):
        # 自定义融合逻辑
        fused = self.custom_fusion(features)
        return fused
```

### 3. 添加新的损失函数

```python
# 在 losses.py 中扩展
class CustomLoss(nn.Module):
    def forward(self, predictions, targets):
        # 实现自定义损失
        return loss

# 在 SetCriterion 中添加
self.losses.append("custom")
def loss_custom(self, outputs, targets, indices, num_boxes):
    return {"loss_custom": self.custom_loss(outputs, targets)}
```

## 预定义配置

```python
from configs.default_config import (
    get_base_config,      # 基础配置
    get_mot17_config,     # MOT17数据集配置
    get_dance_config,     # DanceTrack数据集配置
    get_debug_config      # 调试配置
)

# 获取配置
config = get_mot17_config()
```

## 注意事项

1. **内存使用**：多粒度注意力会同时处理多个分辨率的特征，内存占用较高。建议在训练时使用适当的批次大小。

2. **时序模式**：启用时序建模时，需要准备视频序列数据，输入维度为 `[B, T, C, H, W]`。

3. **预训练权重**：骨干网络支持加载ImageNet预训练权重，建议在 `ResNetBackbone` 中实现加载逻辑。

## 未来改进方向

- [ ] 引入 Deformable Attention 减少计算量
- [ ] 添加更精细的时序对齐机制
- [ ] 实现多尺度查询交互
- [ ] 集成 ReID 特征学习
- [ ] 支持端到端训练的数据加载器

## 许可证

MIT License

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{mg_motrv2,
  title={MG-MOTRv2: Multi-Granularity Multi-Object Tracking with Transformer},
  year={2024}
}
```
