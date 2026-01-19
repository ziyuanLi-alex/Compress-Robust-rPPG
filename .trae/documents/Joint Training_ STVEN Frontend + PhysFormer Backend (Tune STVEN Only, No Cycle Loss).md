## Goal
- Add/finish a joint-training pipeline where STVEN enhances input video frames and a *frozen* pretrained PhysFormer predicts rPPG.
- Inputs: trained PhysFormer weights + pretrained STVEN weights.
- Trainable params: **STVEN only**.
- Loss: **rPPG-only** (no STVEN cycle/reconstruction loss).

## Current State (What Exists / What’s Broken)
- There is already a wrapper model [PhysFormerWithSTVEN](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/model/STVEN.py#L331-L403) and a joint trainer [JointSTVENPhysFormerTrainer.py](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/trainer/JointSTVENPhysFormerTrainer.py).
- Joint trainer already freezes PhysFormer and optimizes only STVEN, and it does not implement cycle loss.
- However, joint training is not currently runnable end-to-end:
  - `main.py` always calls `trainer.test(data_loader_dict)`; joint trainer lacks `test(...)` so it will error.
  - `main.py` has a special-case loader branch for `JointSTPhys` that contains a `pass` and can leave loaders undefined.
  - `joint_st_phys.yaml` contains config keys not defined in [config.py](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/config.py) (e.g., `PRETRAINED_PATH`, lowercase `image_size/patches/...`), which likely breaks YACS merging.
  - STVEN pretrained weights can be incompatible with joint config if `use_bitrate_labels` differs (conv1 input channels mismatch).

## Implementation Plan
### 1) Make config schema accept pretrained weight paths
- Update [config.py](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/config.py) to add:
  - `MODEL.STVEN.PRETRAINED_PATH` (default empty string)
  - `MODEL.PHYSFORMER.PRETRAINED_PATH` (default empty string)
- Keep existing PhysFormer hyperparam keys (`PATCH_SIZE/DIM/FF_DIM/NUM_HEADS/NUM_LAYERS/THETA`) to match the rest of the codebase.

### 2) Refactor PhysFormerWithSTVEN construction to match PhysFormerTrainer
- Update [PhysFormerWithSTVEN](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/model/STVEN.py#L331-L403) so it can construct PhysFormer using the same knobs as [PhysFormerTrainer](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/trainer/PhysFormerTrainer.py):
  - `patches = (config.MODEL.PHYSFORMER.PATCH_SIZE,) * 3`
  - `dropout_rate = config.MODEL.DROP_RATE`
  - `image_size = (chunk_len, H, W)` derived from the training preprocess settings
- Maintain backward compatibility: if the old lowercase keys (`image_size`, `patches`, etc.) exist, prefer them; otherwise use the standard config fields.

### 3) Make pretrained loading robust and explicit
- In [JointSTVENPhysFormerTrainer._load_pretrained_weights](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/trainer/JointSTVENPhysFormerTrainer.py#L58-L88):
  - Support loading STVEN-only checkpoints that may be:
    - plain `state_dict` (`conv1...` keys)
    - wrapper `{'state_dict': ...}`
    - joint wrapper checkpoints that contain `stven.*` keys (strip prefix before loading into `self.model.stven`)
  - Support PhysFormer checkpoints saved under DataParallel (`module.*` prefix stripping).
  - Add a “shape-mismatch tolerant” path for **STVEN conv1 weight** when pretrained used `use_bitrate_labels=True` but joint uses `False` (or vice versa):
    - If checkpoint conv1 has more input channels, slice to expected channels.
    - If it has fewer, pad remaining channels with zeros.
  - For other shape mismatches (base_channels/blocks), fail fast with a clear error message (those represent true architecture mismatch).

### 4) Fix the joint training loop to use PhysFormer’s intended rPPG losses
- Update [JointSTVENPhysFormerTrainer.train](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/trainer/JointSTVENPhysFormerTrainer.py#L89-L165) to mirror PhysFormer training, while only backpropagating into STVEN:
  - Keep PhysFormer frozen (`requires_grad=False`) and put it in `eval()` to disable dropout / BN updates.
  - Use NegPearson + frequency CE + KL via [TorchLossComputer](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/loss/PhysFormerLossComputer.py).
  - Reuse HR extraction from [PhysFormerTrainer.get_hr](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/trainer/PhysFormerTrainer.py#L272-L274).
  - Optimizer should be `Adam(self.model.stven.parameters(), lr=...)` (not `filter(model.parameters())`) to guarantee “STVEN only”.
  - No cycle/reconstruction losses (explicitly remove any remnants; keep rPPG-only objective).

### 5) Implement validation/test for the joint trainer (required by main)
- Add `valid(self, data_loader_dict)` and `test(self, data_loader_dict)` to the joint trainer.
- For validation, follow PhysFormer validation RMSE-on-HR behavior.
- For test, follow PhysFormer’s test pattern:
  - collect per-subject predictions/labels, compute metrics using [calculate_metrics](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/evaluation/metrics.py) if consistent with other trainers, and optionally save outputs via [BaseTrainer.save_test_outputs](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/neural_methods/trainer/BaseTrainer.py#L29-L55).

### 6) Fix data loader selection for JointSTPhys
- Update [main.py](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/main.py) so that:
  - Only STVEN pretraining uses the special `STVENLoader`.
  - JointSTPhys uses the standard dataset loader selection chain (UBFC-rPPG, PURE, etc.), removing the broken `pass` block.
- Optionally add JointSTPhys support to `only_test` mode as well (currently absent).

### 7) Update the joint YAML to be merge-safe and aligned
- Update [joint_st_phys.yaml](file:///home/zyuanli/dev/projects/Compress-Robust-rPPG/configs/train_configs/joint_st_phys.yaml) to:
  - Use `MODEL.DROP_RATE` + `MODEL.PHYSFORMER.{PATCH_SIZE,DIM,FF_DIM,NUM_HEADS,NUM_LAYERS,THETA}` only.
  - Add `MODEL.PHYSFORMER.PRETRAINED_PATH` and `MODEL.STVEN.PRETRAINED_PATH` fields.
  - Ensure `MODEL.STVEN.use_bitrate_labels` matches the pretrained STVEN checkpoint; if you keep it `True`, joint trainer will provide a configurable default bitrate label when the dataset doesn’t provide one.

## Verification Plan (After Implementation)
- Smoke test: instantiate JointSTPhys with the updated config, run a forward pass on a random tensor shaped like `[B,3,T,H,W]`.
- Check gradient flow: confirm PhysFormer params have no grads and STVEN params do.
- Run 1 mini “train” iteration on a tiny DataLoader to