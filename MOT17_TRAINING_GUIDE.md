# MOT17 训练指南

本文档介绍如何在MOT17数据集上训练MG-MOTRv2模型。

## 目录

1. [数据准备](#数据准备)
2. [训练配置](#训练配置)
3. [开始训练](#开始训练)
4. [模型评估](#模型评估)
5. [调参建议](#调参建议)

---

## 数据准备

### 1. 下载MOT17数据集

```bash
# 创建数据目录
mkdir -p data

# 下载MOT17数据集
# 访问 https://motchallenge.net/data/MOT17/ 下载数据集
# 或使用官方脚本:
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip -d data/
```

### 2. 目录结构

确保数据集结构如下：

```
data/MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── img1/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── gt/
│   │       └── gt.txt
│   ├── MOT17-02-FRCNN/
│   ├── MOT17-02-SDP/
│   ├── MOT17-04-DPM/
│   └── ... (共7个序列 x 3个检测器 = 21个训练序列)
└── test/
    └── ... (测试序列)
```

---

## 训练配置

### 默认配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 256 | 特征维度 |
| `num_queries` | 300 | 查询数量 |
| `batch_size` | 2 | 批次大小 |
| `lr` | 2e-4 | 学习率 |
| `lr_backbone` | 2e-5 | 骨干网络学习率 |
| `epochs` | 300 | 训练轮数 |
| `input_size` | (800, 1333) | 输入图像尺寸 |

### 配置文件

可以通过修改 `mg_motrv2/configs/default_config.py` 来调整默认配置，或使用命令行参数覆盖。

---

## 开始训练

### 快速开始

```bash
# 基础训练（使用FRCNN检测器版本）
python -m mg_motrv2.train \
    --dataset_root ./data/MOT17 \
    --batch_size 2 \
    --epochs 300
```

### 调试模式

```bash
# 调试模式（小模型，快速验证代码）
python -m mg_motrv2.train --debug
```

### 高级训练选项

```bash
# 使用所有检测器版本进行数据增强
python -m mg_motrv2.train \
    --dataset_root ./data/MOT17 \
    --use_all_detectors \
    --batch_size 2 \
    --epochs 300 \
    --output_dir ./outputs/mot17_all_detectors

# 调整学习率和批次大小
python -m mg_motrv2.train \
    --dataset_root ./data/MOT17 \
    --batch_size 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --epochs 300

# 调整输入尺寸
python -m mg_motrv2.train \
    --dataset_root ./data/MOT17 \
    --input_height 600 \
    --input_width 1000 \
    --batch_size 4

# 恢复训练
python -m mg_motrv2.train \
    --dataset_root ./data/MOT17 \
    --resume ./outputs/checkpoint_0100.pth
```

### 全部参数

```bash
python -m mg_motrv2.train --help
```

---

## 模型评估

### 生成跟踪结果

```bash
python -m mg_motrv2.eval_mot17 \
    --checkpoint ./outputs/checkpoint_best.pth \
    --dataset_root ./data/MOT17 \
    --output_dir ./eval_results \
    --conf_threshold 0.5
```

### 计算MOTA, IDF1等指标

```bash
# 安装motmetrics
pip install motmetrics

# 计算指标
python -m motmetrics.apps.eval_motchallenge \
    ./data/MOT17/train \
    ./eval_results
```

---

## 调参建议

### GPU显存不足时

```bash
# 减小批次大小和输入尺寸
python -m mg_motrv2.train \
    --batch_size 1 \
    --input_height 600 \
    --input_width 1000
```

### 提高精度

```bash
# 使用更大的输入尺寸
python -m mg_motrv2.train \
    --input_height 1080 \
    --input_width 1920 \
    --batch_size 1
```

### 加速训练

```bash
# 使用所有检测器版本（3倍数据）
python -m mg_motrv2.train \
    --use_all_detectors \
    --num_workers 8
```

### 微调策略

```bash
# 先训练检测部分（冻结骨干网络）
# 然后在configs中设置不同学习率进行微调
python -m mg_motrv2.train \
    --lr 2e-4 \
    --lr_backbone 2e-6 \
    --resume ./pretrained.pth
```

---

## 常见问题

### Q: 找不到数据集
A: 确保 `--dataset_root` 指向正确的MOT17目录，且目录结构正确。

### Q: CUDA out of memory
A: 减小 `--batch_size` 或 `--input_height` / `--input_width`。

### Q: 训练损失不下降
A: 
- 检查学习率是否过大/过小
- 检查数据是否正确加载
- 使用 `--debug` 模式验证代码

### Q: 评估结果为空
A: 
- 降低 `--conf_threshold`（如 0.3）
- 检查模型是否正确加载

---

## 输出文件

训练完成后，输出目录结构：

```
outputs/
├── config.json              # 训练配置
├── checkpoint_best.pth      # 最佳模型
├── checkpoint_0009.pth      # 定期保存的检查点
├── checkpoint_0019.pth
└── ...
```

检查点包含：
- `model`: 模型权重
- `optimizer`: 优化器状态
- `epoch`: 训练轮数
- `config`: 配置信息
- `train_losses`: 训练损失

---

## 参考

- [MOTChallenge](https://motchallenge.net/)
- [MOT17 Paper](https://arxiv.org/abs/1603.00831)
- [MOTRv2 Paper](https://arxiv.org/abs/2211.09791)
