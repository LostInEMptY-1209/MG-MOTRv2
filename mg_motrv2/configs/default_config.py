"""
Default configuration for MG-MOTRv2
MG-MOTRv2 默认配置

快速开始：
    from configs.default_config import get_base_config, get_debug_config
    
    config = get_base_config()  # 标准配置
    config = get_debug_config()  # 调试配置（快速验证）
"""

from typing import Dict, List, Any


class Config:
    """配置类"""
    
    def __init__(self):
        # ==================== 模型配置 ====================
        self.model = {
            "d_model": 256,          # 特征维度
            "num_queries": 300,       # 查询数量
            "num_classes": 1,         # 类别数（跟踪：前景/背景）
            "granularity_levels": ["fine", "medium", "coarse"],
            
            # 骨干网络
            "backbone": {
                "type": "resnet50",
                "fpn_channels": 256,
            },
            
            # DETR配置
            "detr": {
                "n_heads": 8,
                "n_encoder_layers": 6,
                "n_decoder_layers": 6,
                "dropout": 0.1,
                "use_mg_attn": True,    # 使用多粒度注意力
            },
            
            # 跟踪器配置
            "tracker": {
                "max_tracks": 100,
                "track_threshold": 0.5,
                "missed_tolerance": 30,
            },
        }
        
        # ==================== 训练配置 ====================
        self.train = {
            "batch_size": 2,
            "lr": 2e-4,
            "lr_backbone": 2e-5,      # 骨干网络使用较小学习率
            "weight_decay": 1e-4,
            "lr_drop": 200,           # 学习率衰减轮数
            "epochs": 300,
            "clip_max_norm": 0.1,     # 梯度裁剪
            
            # 损失权重
            "loss_weights": {
                "loss_ce": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            }
        }
        
        # ==================== 数据配置 ====================
        self.data = {
            "dataset": "mot17",
            "data_root": "./data",
            "input_size": [800, 1333],  # [H, W]
            "num_workers": 4,
        }
        
        # ==================== 日志配置 ====================
        self.log = {
            "output_dir": "./outputs",
            "checkpoint_interval": 10,
            "log_interval": 50,
            "eval_interval": 5,
        }
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if hasattr(self, key) and isinstance(getattr(self, key), dict):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        config = cls()
        config.update(config_dict)
        return config


def get_base_config() -> Config:
    """获取基础配置"""
    return Config()


def get_debug_config() -> Config:
    """获取调试配置（小批量快速验证）"""
    config = Config()
    config.model["num_queries"] = 10
    config.model["detr"]["n_encoder_layers"] = 2
    config.model["detr"]["n_decoder_layers"] = 2
    config.train["batch_size"] = 1
    config.train["epochs"] = 2
    config.data["num_workers"] = 0
    return config
