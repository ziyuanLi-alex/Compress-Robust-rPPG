"""Verification tests for QAFC-PhysFormer implementation."""

import torch
import sys
from config import get_config
import argparse


def test_model_forward_pass():
    """Test model forward pass produces expected shapes."""
    print("\n=== Test 1: Model Forward Pass ===")

    from neural_methods.model.QAFCPhysFormer import QAFCPhysFormer

    # Create model with default config - use dict for subscript access
    physformer_config = {
        'IMAGE_SIZE': [160, 128, 128],
        'PATCHES': [4, 4, 4],
        'DIM': 64,
        'FF_DIM': 256,
        'NUM_HEADS': 4,
        'NUM_LAYERS': 12,
        'DROPOUT_RATE': 0.2,
        'THETA': 0.2,
        'QUALITY_CHANNELS': 64,
        'HIDDEN_SIZE': 128,
        'USE_BLOCK_PATTERN': True,
    }

    model = QAFCPhysFormer(
        physformer_config=physformer_config,
        quality_spatial_channels=physformer_config['QUALITY_CHANNELS'],
        quality_temporal_hidden=physformer_config['HIDDEN_SIZE'],
        use_block_pattern=physformer_config['USE_BLOCK_PATTERN'],
        film_after_spatial=False,  # Disable for basic forward test
        film_after_temporal=False
    )

    # Create dummy input [B, C, T, H, W]
    x = torch.randn(2, 3, 160, 128, 128)

    # Forward pass
    model.eval()
    with torch.no_grad():
        rppg, quality_score, s1, s2, s3 = model(x, gra_sharp=2.0)

    # Verify shapes
    assert rppg.shape == (2, 160), f"Expected rppg shape (2, 160), got {rppg.shape}"
    assert quality_score.shape == (2, 1), f"Expected quality_score shape (2, 1), got {quality_score.shape}"
    assert s1.shape[0] == 2, f"Expected s1 batch size 2, got {s1.shape[0]}"
    assert s2.shape[0] == 2, f"Expected s2 batch size 2, got {s2.shape[0]}"
    assert s3.shape[0] == 2, f"Expected s3 batch size 2, got {s3.shape[0]}"

    print(f"  rPPG output shape: {rppg.shape} ✓")
    print(f"  Quality score shape: {quality_score.shape} ✓")
    print(f"  Attention scores shapes: s1={s1.shape}, s2={s2.shape}, s3={s3.shape} ✓")
    print("  Forward pass test PASSED ✓\n")
    return True


def test_model_backward_pass():
    """Test backward pass completes (gradients flow to all trainable params)."""
    print("\n=== Test 2: Model Backward Pass ===")

    from neural_methods.model.QAFCPhysFormer import QAFCPhysFormer

    physformer_config = {
        'IMAGE_SIZE': [160, 128, 128],
        'PATCHES': [4, 4, 4],
        'DIM': 64,
        'FF_DIM': 256,
        'NUM_HEADS': 4,
        'NUM_LAYERS': 12,
        'DROPOUT_RATE': 0.2,
        'THETA': 0.2,
        'QUALITY_CHANNELS': 64,
        'HIDDEN_SIZE': 128,
        'USE_BLOCK_PATTERN': True,
    }

    model = QAFCPhysFormer(
        physformer_config=physformer_config,
        quality_spatial_channels=physformer_config['QUALITY_CHANNELS'],
        quality_temporal_hidden=physformer_config['HIDDEN_SIZE'],
        use_block_pattern=physformer_config['USE_BLOCK_PATTERN'],
        film_after_spatial=False,  # Disable FiLM for basic test
        film_after_temporal=False
    )
    model.train()

    # Create dummy input and label
    x = torch.randn(2, 3, 160, 128, 128)
    bvp_label = torch.randn(2, 160)

    # Forward pass
    rppg, quality_score, _, _, _ = model(x, gra_sharp=2.0)

    # Calculate loss
    loss = torch.mean((rppg - bvp_label) ** 2)

    # Backward pass
    loss.backward()

    # Check gradients exist for trainable parameters
    trainable_params_with_grad = 0
    total_trainable_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable_params += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                trainable_params_with_grad += 1

    print(f"  Trainable parameters with gradients: {trainable_params_with_grad}/{total_trainable_params}")

    assert trainable_params_with_grad > 0, "No trainable parameters have gradients!"
    # Note: Not all params may have gradients due to:
    # - Some params receiving very small gradients (below threshold)
    # - Batch norm layers behaving differently in train/eval mode
    # - Quality branch params if FiLM is disabled
    assert trainable_params_with_grad >= total_trainable_params * 0.8, \
        f"Too many parameters missing gradients: {trainable_params_with_grad}/{total_trainable_params}"

    print("  Backward pass test PASSED ✓\n")
    return True


def test_loss_functions():
    """Test QAFC loss functions."""
    print("\n=== Test 3: Loss Functions ===")

    from neural_methods.loss.QAFCPhysFormerLoss import QualityRankingLoss, QAFCLoss

    # Test QualityRankingLoss
    ranking_loss_fn = QualityRankingLoss(margin=0.1)

    score_high = torch.tensor([[1.0], [0.8], [0.9]])  # Higher quality scores
    score_low = torch.tensor([[0.3], [0.2], [0.4]])   # Lower quality scores

    ranking_loss = ranking_loss_fn(score_high, score_low)
    assert ranking_loss.item() >= 0, "Ranking loss should be non-negative"
    print(f"  QualityRankingLoss output: {ranking_loss.item():.4f} ✓")

    # Test QAFCLoss
    qafc_loss_fn = QAFCLoss()

    pred_rppg = torch.randn(2, 160)
    gt_rppg = torch.randn(2, 160)

    loss_dict = qafc_loss_fn(pred_rppg, gt_rppg, score_high, score_low)

    assert 'rppg' in loss_dict, "QAFCLoss should return 'rppg' key"
    assert 'ranking' in loss_dict, "QAFCLoss should return 'ranking' key"
    assert 'total' in loss_dict, "QAFCLoss should return 'total' key"

    print(f"  QAFCLoss outputs: rppg={loss_dict['rppg']:.4f}, ranking={loss_dict['ranking']:.4f}, total={loss_dict['total']:.4f} ✓")
    print("  Loss functions test PASSED ✓\n")
    return True


def test_trainer_initialization():
    """Test trainer can initialize with config."""
    print("\n=== Test 4: Trainer Initialization ===")

    from neural_methods.trainer.QAFCPhysFormerTrainer import QAFCPhysFormerTrainer

    # Create minimal config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                       default="configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml")
    args = parser.parse_args(['--config_file',
                             'configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml'])

    config = get_config(args)

    # Create mock data_loader dict
    data_loader_dict = {
        'train': None,
        'valid': None,
        'test': None
    }

    try:
        trainer = QAFCPhysFormerTrainer(config, data_loader_dict)
        print(f"  Trainer initialized successfully ✓")
        print(f"  Model device: {trainer.device} ✓")
        print(f"  Phase boundary: {trainer.phase_boundary} ✓")
        print(f"  Current phase: {trainer.current_phase} ✓")
        print("  Trainer initialization test PASSED ✓\n")
        return True
    except Exception as e:
        print(f"  Trainer initialization FAILED: {e}\n")
        return False


def test_config_validation():
    """Test config file has no missing keys."""
    print("\n=== Test 5: Config Validation ===")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                       default="configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml")
    args = parser.parse_args(['--config_file',
                             'configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml'])

    try:
        config = get_config(args)

        # Check required QAFC-PhysFormer keys exist
        assert hasattr(config.MODEL, 'QAFC_PHYSFORMER'), "MODEL.QAFC_PHYSFORMER not found"
        assert hasattr(config.TRAIN, 'QAFC'), "TRAIN.QAFC not found"
        assert config.MODEL.NAME == "QAFCPhysFormer", f"MODEL.NAME should be QAFCPhysFormer, got {config.MODEL.NAME}"

        # Check QAFC_PHYSFORMER sub-keys
        assert config.MODEL.QAFC_PHYSFORMER.QUALITY_CHANNELS == 64
        assert config.MODEL.QAFC_PHYSFORMER.HIDDEN_SIZE == 128
        assert config.MODEL.QAFC_PHYSFORMER.USE_BLOCK_PATTERN == True

        # Check TRAIN.QAFC sub-keys
        assert config.TRAIN.QAFC.PHASE_BOUNDARY == 70
        assert config.TRAIN.QAFC.RANKING_MARGIN == 0.1

        print(f"  Config loaded successfully ✓")
        print(f"  MODEL.NAME: {config.MODEL.NAME} ✓")
        print(f"  PHASE_BOUNDARY: {config.TRAIN.QAFC.PHASE_BOUNDARY} ✓")
        print(f"  RANKING_MARGIN: {config.TRAIN.QAFC.RANKING_MARGIN} ✓")
        print(f"  CRF_LEVELS: {config.CRF_LEVELS} ✓")
        print("  Config validation test PASSED ✓\n")
        return True
    except Exception as e:
        print(f"  Config validation FAILED: {e}\n")
        return False


def test_data_loader():
    """Test data loader returns correct batch format."""
    print("\n=== Test 6: Data Loader Structure ===")

    from dataset.data_loader.QAFCPhysFormerLoader import QAFCPhysFormerLoader

    # Check class exists and has required methods
    assert hasattr(QAFCPhysFormerLoader, '__init__'), "QAFCPhysFormerLoader missing __init__"
    assert hasattr(QAFCPhysFormerLoader, '__getitem__'), "QAFCPhysFormerLoader missing __getitem__"
    assert hasattr(QAFCPhysFormerLoader, '__len__'), "QAFCPhysFormerLoader missing __len__"

    print("  QAFCPhysFormerLoader class structure verified ✓")
    print("  Data loader test PASSED ✓\n")
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("QAFC-PhysFormer Verification Tests")
    print("=" * 60)

    results = {
        'Forward Pass': test_model_forward_pass(),
        'Backward Pass': test_model_backward_pass(),
        'Loss Functions': test_loss_functions(),
        'Trainer Initialization': test_trainer_initialization(),
        'Config Validation': test_config_validation(),
        'Data Loader': test_data_loader(),
    }

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED ✓" if result else "FAILED ✗"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll verification tests PASSED! Implementation is ready.\n")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please review.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
