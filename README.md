# MG-MOTRv2: Multi-Granularity Multi-Object Tracking

基于 MOTRv2 架构的多粒度注意力机制多目标跟踪研究框架。

## 核心创新

- **多粒度注意力机制**：整合细粒度(fine)、中粒度(medium)、粗粒度(coarse)三个空间层级的特征
- **自适应融合**：根据输入动态调整不同粒度的权重
- **模块化设计**：易于扩展和实验

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行演示

```bash
# 运行所有演示
python -m mg_motrv2.demo --all

# 特定演示
python -m mg_motrv2.demo --mg-attn    # 多粒度注意力
python -m mg_motrv2.demo --backbone   # 骨干网络
python -m mg_motrv2.demo --model      # 完整模型
```

### 运行测试

```bash
python test_basic.py
```

## 项目结构

```
mg_motrv2/
├── models/
│   ├── mg_attention.py      # 多粒度注意力核心
│   ├── backbone.py          # 多粒度骨干网络
│   ├── mg_motr.py          # MG-DETR头
│   └── mg_motrv2.py        # 完整模型
├── utils/
│   ├── matcher.py          # 匈牙利匹配器
│   └── losses.py           # 损失函数
├── configs/
│   └── default_config.py   # 默认配置
├── train.py                # 训练脚本
└── demo.py                 # 演示脚本
```

## 核心模块使用

### 多粒度注意力

```python
from mg_motrv2.models import MultiGranularityAttention

mg_attn = MultiGranularityAttention(
    d_model=256,
    n_heads=8,
    n_levels=3,            # fine, medium, coarse
    fusion_type="adaptive" # adaptive/concat/sum
)

features = [fine_feat, medium_feat, coarse_feat]
fused = mg_attn(features)  # [B, N, C]
```

### 完整模型

```python
from mg_motrv2.models.mg_motrv2 import build_model
from mg_motrv2.configs.default_config import Config

config = Config()
model = build_model(config.to_dict()['model'])

outputs = model(images)  # 检测和跟踪结果
```

## 扩展接口

### 1. 添加新粒度级别

```python
# 在 GranularityLevel 中扩展 stride 字典
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

### 3. 添加新损失函数

```python
# 在 SetCriterion 中添加
self.losses.append("custom")
def loss_custom(self, outputs, targets, indices, num_boxes):
    return {"loss_custom": custom_loss(outputs, targets)}
```

## 预定义配置

```python
from mg_motrv2.configs.default_config import (
    get_base_config,    # 基础配置
    get_debug_config,   # 调试配置（快速验证）
)

config = get_base_config()
```

## 预留改进空间

1. **骨干网络预训练权重加载**
2. **数据加载器实现**
3. **ReID特征学习**
4. **Deformable Attention**（性能优化）
5. **更精细的时序对齐**
6. **评估指标计算**（MOTA, IDF1等）

## 许可证

MIT License
