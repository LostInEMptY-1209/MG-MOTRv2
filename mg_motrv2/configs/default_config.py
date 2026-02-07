"""
Default configuration for MG-MOTRv2
MG-MOTRv2 默认配置
"""

from typing import Dict, List, Any


class Config:
    """配置类"""
    
    def __init__(self):
        # ==================== 模型配置 ====================
        self.model = {
            # 基础维度
            "d_model": 256,
            
            # 查询数量
            "num_queries": 300,
            
            # 类别数（跟踪通常只需要前景/背景，设为1）
            "num_classes": 1,
            
            # 时序帧数
            "n_frames": 1,
            
            # 是否使用时序建模
            "use_temporal": False,
            
            # 多粒度级别
            "granularity_levels": ["fine", "medium", "coarse"],
            
            # 骨干网络配置
            "backbone": {
                "type": "resnet50",
                "fpn_channels": 256,
                "frozen_stages": 1,  # 冻结前几个stage
            },
            
            # DETR配置
            "detr": {
                "n_heads": 8,
                "n_encoder_layers": 6,
                "n_decoder_layers": 6,
                "dropout": 0.1,
                "use_mg_attn": True,
            },
            
            # 跟踪器配置
            "tracker": {
                "max_tracks": 100,
                "track_threshold": 0.5,
                "missed_tolerance": 30,  # 最大容忍丢失帧数
            },
            
            # 多粒度注意力配置
            "mg_attention": {
                "n_heads": 8,
                "fusion_type": "adaptive",  # "adaptive", "concat", "sum"
                "dropout": 0.1,
            }
        }
        
        # ==================== 训练配置 ====================
        self.train = {
            # 批次大小
            "batch_size": 2,
            
            # 学习率
            "lr": 2e-4,
            "lr_backbone": 2e-5,  # 骨干网络使用较小学习率
            "weight_decay": 1e-4,
            
            # 学习率调度
            "lr_drop": 200,  # 在第200个epoch降低学习率
            
            # 训练周期
            "epochs": 300,
            
            # 梯度裁剪
            "clip_max_norm": 0.1,
            
            # 优化器
            "optimizer": "adamw",
            
            # 损失权重
            "loss_weights": {
                "loss_ce": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            }
        }
        
        # ==================== 数据配置 ====================
        self.data = {
            # 数据集名称
            "dataset": "mot17",  # "mot17", "mot20", "dancetrack"
            
            # 数据路径
            "data_root": "./data",
            
            # 输入图像尺寸
            "input_size": [800, 1333],  # [H, W]
            
            # 数据增强
            "augmentation": {
                "random_flip": True,
                "random_crop": True,
                "color_jitter": 0.4,
                "random_erase": 0.5,
            },
            
            # 采样
            "sampler_steps": [],  # 在哪个epoch改变采样策略
            "sampler_lengths": [],  # 对应的采样长度
            
            # 数据加载
            "num_workers": 4,
            "pin_memory": True,
        }
        
        # ==================== 测试配置 ====================
        self.test = {
            # 测试时增强
            "tta": False,
            
            # 置信度阈值
            "score_threshold": 0.5,
            
            # NMS阈值
            "nms_threshold": 0.5,
            
            # 最大检测数
            "max_detections": 300,
        }
        
        # ==================== 日志配置 ====================
        self.log = {
            # 输出目录
            "output_dir": "./outputs",
            
            # 检查点保存频率
            "checkpoint_interval": 10,  # 每10个epoch保存一次
            
            # 日志频率
            "log_interval": 50,  # 每50次迭代记录一次
            
            # 评估频率
            "eval_interval": 5,  # 每5个epoch评估一次
            
            # TensorBoard
            "use_tensorboard": True,
            
            # Wandb
            "use_wandb": False,
        }
        
        # ==================== 硬件配置 ====================
        self.device = {
            # 使用GPU
            "use_cuda": True,
            
            # GPU ID
            "gpu_ids": [0],
            
            # 分布式训练
            "distributed": False,
            "world_size": 1,
            "dist_url": "env://",
        }
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict) and isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        config = cls()
        config.update(config_dict)
        return config


# 预定义配置
def get_base_config() -> Config:
    """获取基础配置"""
    return Config()


def get_mot17_config() -> Config:
    """获取MOT17数据集配置"""
    config = Config()
    config.data["dataset"] = "mot17"
    config.data["input_size"] = [800, 1333]
    config.train["epochs"] = 200
    config.train["lr_drop"] = 150
    return config


def get_dance_config() -> Config:
    """获取DanceTrack数据集配置"""
    config = Config()
    config.data["dataset"] = "dancetrack"
    config.data["input_size"] = [800, 1333]
    config.train["epochs"] = 100
    config.train["lr_drop"] = 80
    # DanceTrack需要更强的时序建模
    config.model["use_temporal"] = True
    config.model["n_frames"] = 4
    return config


def get_debug_config() -> Config:
    """获取调试配置（小批量快速测试）"""
    config = Config()
    config.model["num_queries"] = 10
    config.model["detr"]["n_encoder_layers"] = 2
    config.model["detr"]["n_decoder_layers"] = 2
    config.train["batch_size"] = 1
    config.train["epochs"] = 2
    config.data["num_workers"] = 0
    return config
