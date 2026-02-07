"""
Basic tests for MG-MOTRv2
基础测试 - 验证核心模块可以正常运行

运行：
    python test_basic.py
"""

import torch
import sys


def test_granularity_level():
    """测试 GranularityLevel"""
    from mg_motrv2.models.mg_attention import GranularityLevel
    
    layer = GranularityLevel(d_model=256, n_heads=8, granularity="fine")
    x = torch.randn(2, 100, 256)
    out = layer(x)
    
    assert out.shape == x.shape
    print("✓ GranularityLevel test passed")


def test_multi_granularity_attention():
    """测试 MultiGranularityAttention"""
    from mg_motrv2.models.mg_attention import MultiGranularityAttention
    
    mg_attn = MultiGranularityAttention(d_model=256, n_heads=8, n_levels=3)
    features = [torch.randn(2, 100, 256) for _ in range(3)]
    
    fused = mg_attn(features)
    assert fused.shape == (2, 100, 256)
    print("✓ MultiGranularityAttention test passed")


def test_temporal_granularity_attention():
    """测试 TemporalGranularityAttention"""
    from mg_motrv2.models.mg_attention import TemporalGranularityAttention
    
    temporal_attn = TemporalGranularityAttention(
        d_model=256, n_heads=8, n_frames=4, n_levels=3
    )
    
    frame_features = [
        [torch.randn(2, 100, 256) for _ in range(3)]
        for _ in range(4)
    ]
    
    output = temporal_attn(frame_features)
    assert output.shape == (2, 100, 256)
    print("✓ TemporalGranularityAttention test passed")


def test_backbone():
    """测试 MultiGranularityBackbone"""
    from mg_motrv2.models.backbone import MultiGranularityBackbone
    
    backbone = MultiGranularityBackbone(
        backbone_type="resnet50",
        fpn_channels=256,
        granularity_levels=["fine", "medium", "coarse"]
    )
    
    images = torch.randn(2, 3, 800, 1333)
    features = backbone(images)
    
    assert len(features) == 3
    assert all(level in features for level in ["fine", "medium", "coarse"])
    print("✓ MultiGranularityBackbone test passed")


def test_mg_detr_head():
    """测试 MG_DETRHead"""
    from mg_motrv2.models.mg_motr import MG_DETRHead
    
    detr_head = MG_DETRHead(
        d_model=256,
        num_classes=1,
        num_queries=100,
        use_mg_attn=True,
        n_granularity_levels=3
    )
    
    feat = torch.randn(2, 256, 25, 40)
    mg_features = [
        torch.randn(2, 256, 50, 80),
        torch.randn(2, 256, 25, 40),
        torch.randn(2, 256, 13, 20)
    ]
    
    outputs = detr_head(feat, mg_features)
    
    assert "pred_logits" in outputs
    assert "pred_boxes" in outputs
    assert outputs["pred_logits"].shape == (2, 100, 2)  # 1 class + 1 background
    assert outputs["pred_boxes"].shape == (2, 100, 4)
    print("✓ MG_DETRHead test passed")


def test_full_model():
    """测试完整模型"""
    from mg_motrv2.models.mg_motrv2 import MGMOTRv2
    from mg_motrv2.configs.default_config import Config
    
    config = Config()
    config_dict = config.to_dict()
    
    # 使用较小配置
    config_dict['model']['num_queries'] = 10
    config_dict['model']['detr']['n_encoder_layers'] = 2
    config_dict['model']['detr']['n_decoder_layers'] = 2
    
    model = MGMOTRv2(config_dict['model'])
    model.eval()
    
    images = torch.randn(1, 3, 400, 600)
    
    with torch.no_grad():
        outputs = model(images)
    
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    print("✓ Full model test passed")


def test_matcher():
    """测试 HungarianMatcher"""
    from mg_motrv2.utils.matcher import HungarianMatcher
    
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    
    outputs = {
        "pred_logits": torch.randn(2, 100, 2),
        "pred_boxes": torch.rand(2, 100, 4).sigmoid()
    }
    
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)},
        {"labels": torch.tensor([0, 0]), "boxes": torch.rand(2, 4)}
    ]
    
    indices = matcher(outputs, targets)
    assert len(indices) == 2
    print("✓ HungarianMatcher test passed")


def test_losses():
    """测试 SetCriterion"""
    from mg_motrv2.utils.losses import SetCriterion
    from mg_motrv2.utils.matcher import HungarianMatcher
    
    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
        losses=["labels", "boxes"]
    )
    
    outputs = {
        "pred_logits": torch.randn(2, 10, 2),
        "pred_boxes": torch.rand(2, 10, 4).sigmoid()
    }
    
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)},
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)}
    ]
    
    losses = criterion(outputs, targets)
    assert "loss_ce" in losses
    assert "loss_bbox" in losses
    assert "loss_giou" in losses
    print("✓ SetCriterion test passed")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Running MG-MOTRv2 Basic Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_granularity_level,
        test_multi_granularity_attention,
        test_temporal_granularity_attention,
        test_backbone,
        test_mg_detr_head,
        test_full_model,
        test_matcher,
        test_losses,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
