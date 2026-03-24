# rPPG-Toolbox Training and Evaluation Pipeline Documentation

**Purpose:** This document provides comprehensive knowledge of the rPPG-Toolbox training and evaluation pipeline to facilitate integrating new architectures (e.g., QAFC-PhysFormer) into the framework.

**Date:** 2026-03-24

---

## Table of Contents

1. [Overview](#1-overview)
2. [Entry Point and Execution Flow](#2-entry-point-and-execution-flow)
3. [Configuration System](#3-configuration-system)
4. [Data Loading Pipeline](#4-data-loading-pipeline)
5. [Model Architecture Pattern](#5-model-architecture-pattern)
6. [Trainer Pattern](#6-trainer-pattern)
7. [Training Loop](#7-training-loop)
8. [Validation and Model Selection](#8-validation-and-model-selection)
9. [Testing and Evaluation](#9-testing-and-evaluation)
10. [Loss Functions](#10-loss-functions)
11. [Metrics](#11-metrics)
12. [Integration Guide for New Architectures](#12-integration-guide-for-new-architectures)

---

## 1. Overview

### 1.1 Toolbox Modes

The framework supports three operational modes configured via `TOOLBOX_MODE`:

| Mode | Description | Use Case |
|------|-------------|----------|
| `train_and_test` | Full training + evaluation pipeline | Training new models |
| `only_test` | Evaluation with pretrained weights | Inference/benchmarking |
| `unsupervised_method` | Traditional signal processing (POS, CHROM, ICA, etc.) | Baseline comparison |

### 1.2 Architecture

```
rPPG-Toolbox/
├── main.py                    # Entry point - routes to train/test/unsupervised
├── config.py                  # YACS config system
├── configs/
│   ├── train_configs/         # [TRAIN]_[VALID]_[TEST]_[MODEL].yaml
│   └── infer_configs/         # [TRAIN]_[TEST]_[MODEL].yaml
├── dataset/
│   └── data_loader/           # Dataset loaders (BaseLoader + dataset-specific)
├── neural_methods/
│   ├── model/                 # Network architectures
│   ├── trainer/               # Training loops (one Trainer per model)
│   └── loss/                  # Loss functions
├── unsupervised_methods/      # Traditional signal processing
├── evaluation/                # Metrics (MAE, RMSE, MAPE, Pearson, SNR, BA)
└── tools/                     # Visualization utilities
```

---

## 2. Entry Point and Execution Flow

### 2.1 Main Function (`main.py`)

The entry point performs the following steps:

```python
# 1. Parse arguments
parser = argparse.ArgumentParser()
parser = add_args(parser)  # Adds --config_file
parser = BaseTrainer.add_trainer_args(parser)
parser = BaseLoader.add_data_loader_args(parser)
args = parser.parse_args()

# 2. Load configuration
config = get_config(args)

# 3. Create data loaders based on TOOLBOX_MODE
data_loader_dict = dict()

# For train_and_test mode:
if config.TOOLBOX_MODE == "train_and_test":
    # Create train_loader
    train_loader = <dataset-specific-loader>(
        name="train",
        data_path=config.TRAIN.DATA.DATA_PATH,
        config_data=config.TRAIN.DATA,
        device=config.DEVICE
    )
    data_loader_dict['train'] = DataLoader(
        dataset=train_loader,
        num_workers=16,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_generator
    )

    # Create valid_loader
    valid_loader = <dataset-specific-loader>(...)
    data_loader_dict['valid'] = DataLoader(...)

    # Create test_loader
    test_loader = <dataset-specific-loader>(...)
    data_loader_dict['test'] = DataLoader(...)

# 4. Route to appropriate function
if config.TOOLBOX_MODE == "train_and_test":
    train_and_test(config, data_loader_dict)
elif config.TOOLBOX_MODE == "only_test":
    test(config, data_loader_dict)
elif config.TOOLBOX_MODE == "unsupervised_method":
    unsupervised_method_inference(config, data_loader_dict)
```

### 2.2 Model Dispatch

Models are instantiated via a factory pattern in `train_and_test()` and `test()`:

```python
def train_and_test(config, data_loader_dict):
    if config.MODEL.NAME == "Physnet":
        model_trainer = PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "PhysFormer":
        model_trainer = PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "JointSTPhys":
        model_trainer = JointSTVENPhysFormerTrainer.JointSTVENPhysFormerTrainer(config, data_loader_dict)
    # ... other models
    else:
        raise ValueError('Your Model is Not Supported Yet!')

    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)
```

### 2.3 Loader Dispatch

Data loaders are selected based on model type and dataset:

```python
# Special loader for STVEN/JointSTPhys models
if config.MODEL.NAME == "STVEN" or config.MODEL.NAME == "JointSTPhys":
    train_loader = data_loader.STVENLoader.STVENLoader
elif config.TRAIN.DATA.DATASET == "UBFC-rPPG":
    train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
# ... other datasets
```

---

## 3. Configuration System

### 3.1 YACS Configuration

The framework uses YACS (Yet Another Configuration System) for hierarchical configuration management.

**Base Config Structure (`config.py`):**

```python
_C = CN()
_C.TOOLBOX_MODE = ""
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.LR = 1e-4
_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.FS = 30  # Frame rate
_C.TRAIN.DATA.DATASET = ""
_C.TRAIN.DATA.DATA_PATH = ""
_C.TRAIN.DATA.CACHED_PATH = ""
_C.TRAIN.DATA.BEGIN = 0.0  # Data split start
_C.TRAIN.DATA.END = 1.0    # Data split end
_C.TRAIN.DATA.PREPROCESS = CN()
_C.TRAIN.DATA.PREPROCESS.DATA_TYPE = ["DiffNormalized"]
_C.TRAIN.DATA.PREPROCESS.LABEL_TYPE = "DiffNormalized"
_C.TRAIN.DATA.PREPROCESS.DO_CHUNK = True
_C.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH = 160
_C.TRAIN.DATA.PREPROCESS.CROP_FACE = CN()
# ... more preprocessing options

_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.MODEL_DIR = "PreTrainedModels"
_C.MODEL.PHYSFORMER = CN()  # Model-specific configs
_C.MODEL.STVEN = CN()

_C.TEST = CN()
_C.TEST.METRICS = ["MAE", "RMSE", "MAPE", "Pearson"]
_C.TEST.USE_LAST_EPOCH = False
_C.TEST.DATA = CN()

_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.MODEL_PATH = ""
```

### 3.2 Config File Naming Convention

Training configs follow the pattern: `[TRAIN]_[VALID]_[TEST]_[MODEL].yaml`

Example: `PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml`
- Train on PURE (80%)
- Validate on PURE (20%)
- Test on UBFC-rPPG

### 3.3 Key Configuration Parameters

#### Data Splitting
```yaml
TRAIN:
  DATA:
    BEGIN: 0.0
    END: 0.8   # First 80% for training

VALID:
  DATA:
    BEGIN: 0.8
    END: 0.9   # Next 10% for validation

TEST:
  DATA:
    BEGIN: 0.9
    END: 1.0   # Last 10% for testing
```

#### Preprocessing Options
```yaml
PREPROCESS:
  DATA_TYPE: ["DiffNormalized"]  # or "Raw", "Standardized"
  LABEL_TYPE: "DiffNormalized"   # must match DATA_TYPE
  DO_CHUNK: True
  CHUNK_LENGTH: 160              # frames per clip
  CROP_FACE:
    DO_CROP_FACE: True
    BACKEND: "HC"                # "HC" (Haar Cascade) or "Y5F" (YOLO5Face)
    USE_LARGE_FACE_BOX: True
    LARGE_BOX_COEF: 1.5
    DETECTION:
      DO_DYNAMIC_DETECTION: False
      DYNAMIC_DETECTION_FREQUENCY: 30
      USE_MEDIAN_FACE_BOX: False
  RESIZE:
    H: 128
    W: 128
```

#### Model-Specific Configs
```yaml
MODEL:
  NAME: "JointSTPhys"
  DROP_RATE: 0.2
  STVEN:
    PRETRAINED_PATH: "path/to/stven.pth"
    in_channels: 3
    out_channels: 3
    base_channels: 16
    num_st_blocks: 6
    frame_length: 160
    use_bitrate_labels: True
    num_bitrate_levels: 3
  PHYSFORMER:
    PRETRAINED_PATH: "path/to/physformer.pth"
    PATCH_SIZE: 4
    DIM: 96
    FF_DIM: 144
    NUM_HEADS: 4
    NUM_LAYERS: 12
    THETA: 0.7
    IMAGE_SIZE: [160, 128, 128]
    PATCHES: [4, 4, 4]
    DROPOUT_RATE: 0.2
```

---

## 4. Data Loading Pipeline

### 4.1 BaseLoader Class

All dataset loaders inherit from `BaseLoader` which extends PyTorch's `Dataset`.

**Key Responsibilities:**
1. Raw data discovery (`get_raw_data()`)
2. Data splitting (`split_raw_data()`)
3. Preprocessing (`preprocess()`)
4. File list management (`build_file_list()`, `load_preprocessed_data()`)
5. Data iteration (`__getitem__()`, `__len__()`)

### 4.2 Preprocessing Pipeline

```
Raw Video → Face Detection → Resize → Data Transformation → Chunking → .npy files
     ↓
Raw BVP  → Label Transformation → Chunking → .npy files
```

**Preprocessing Steps:**

1. **Face Detection & Cropping**
   - Backend: Haar Cascade (HC) or YOLO5Face (Y5F)
   - Optional: Dynamic detection (every N frames)
   - Optional: Large face box (scale by `LARGE_BOX_COEF`)

2. **Resize**
   - Default: 128x128

3. **Data Transformation**
   - `Raw`: No transformation
   - `DiffNormalized`: Temporal differencing + normalization
   - `Standardized`: Z-score normalization

4. **Label Transformation**
   - Must match `DATA_TYPE`

5. **Chunking**
   - Split into clips of `CHUNK_LENGTH` frames

### 4.3 Preprocessed Data Format

Files are saved as `.npy` arrays:
- Video clips: `{subject}_{chunk}_input{n}.npy` shape `(T, H, W, 3)`
- Label clips: `{subject}_{chunk}_label{n}.npy` shape `(T,)`

Data format options:
- `NDCHW`: (N, D, C, H, W) - batch, time, channels, height, width
- `NCDHW`: (N, C, D, H, W) - batch, channels, time, height, width
- `NDHWC`: (N, D, H, W, C) - batch, time, height, width, channels

### 4.4 File List Management

A CSV file (`FILE_LIST_PATH`) tracks preprocessed files:
```csv
input_files
/home/data/PURE_.../501_input0.npy
/home/data/PURE_.../501_input1.npy
...
```

### 4.5 STVENLoader (Special Case)

For compression-aware models, `STVENLoader` handles pairs of compressed/uncompressed videos:

```python
def __getitem__(self, index):
    # Load compressed data
    compressed_data = np.load(self.inputs[index])

    # Parse CRF level from filename (e.g., "subject1_crf23_input0.npy")
    current_crf = int(re.search(r'_crf(\d+)_', filename).group(1))

    # Load uncompressed pair (CRF 0)
    uncompressed_path = path.replace(f"_crf{current_crf}_", "_crf0_")
    uncompressed_data = np.load(uncompressed_path)

    # Generate one-hot CRF label
    crf_levels = self.config_data.CRF_LEVELS  # [0, 5, 10]
    label_idx = crf_levels.index(current_crf)
    bitrate_label = np.zeros(len(crf_levels))
    bitrate_label[label_idx] = 1.0

    # Load BVP label
    bvp_label = np.load(self.labels[index])

    return compressed_data, uncompressed_data, bitrate_label, bvp_label
```

### 4.6 DataLoader Configuration

```python
# In main.py
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

data_loader_dict['train'] = DataLoader(
    dataset=train_data_loader,
    num_workers=16,
    batch_size=config.TRAIN.BATCH_SIZE,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=train_generator
)
```

---

## 5. Model Architecture Pattern

### 5.1 Model Organization

Models are organized as:
```
neural_methods/
├── model/
│   ├── PhysFormer.py
│   ├── STVEN.py
│   ├── DeepPhys.py
│   └── ...
├── trainer/
│   ├── PhysFormerTrainer.py
│   ├── STVENTrainer.py
│   └── ...
└── loss/
    ├── NegPearsonLoss.py
    └── PhysFormerLossComputer.py
```

### 5.2 Model Base Pattern

Models are plain `nn.Module` subclasses:

```python
class PhysFormer(nn.Module):
    def __init__(self, image_size, patches, dim, ff_dim, num_heads,
                 num_layers, dropout_rate, theta):
        super().__init__()
        # Initialize layers

    def forward(self, x, gra_sharp=2.0):
        # Forward pass
        return rPPG, score1, score2, score3
```

### 5.3 Composite Models (PhysFormerWithSTVEN)

For multi-component architectures:

```python
class PhysFormerWithSTVEN(nn.Module):
    def __init__(self, stven_config, physformer_config):
        super().__init__()
        self.stven = STVEN(**stven_config)
        self.physformer = ViT_ST_ST_Compact3_TDC_gra_sharp(**physformer_config)

    def forward(self, x, bitrate_label=None, gra_sharp=2.0):
        enhanced_video = self.stven(x, bitrate_label)
        rPPG, score1, score2, score3 = self.physformer(enhanced_video, gra_sharp)
        return rPPG, score1, score2, score3
```

---

## 6. Trainer Pattern

### 6.1 BaseTrainer

```python
class BaseTrainer:
    @staticmethod
    def add_trainer_args(parser):
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=float)
        return parser

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass

    def save_test_outputs(self, predictions, labels, config):
        # Saves outputs as pickle
        pass

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):
        # Saves loss/LR plots
        pass
```

### 6.2 Trainer Implementation Pattern

```python
class JointSTVENPhysFormerTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.config = config

        # Initialize model
        self.model = PhysFormerWithSTVEN(
            stven_config=config.MODEL.STVEN,
            physformer_config=config.MODEL.PHYSFORMER
        ).to(self.device)

        # Load pretrained weights
        self._load_pretrained_weights(config)

        # Freeze specific components (e.g., PhysFormer)
        for param in self.model.physformer.parameters():
            param.requires_grad = False

        # Optimizer (only trainable params)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.TRAIN.LR
        )

        # Loss functions
        self.criterion_Pearson = Neg_Pearson()

        # Tracking
        self.min_valid_loss = None
        self.best_epoch = 0

    def _load_pretrained_weights(self, config):
        """Load pretrained weights for components."""
        if config.MODEL.STVEN.PRETRAINED_PATH:
            stven_state = torch.load(config.MODEL.STVEN.PRETRAINED_PATH)
            if 'state_dict' in stven_state:
                stven_state = stven_state['state_dict']
            self.model.stven.load_state_dict(stven_state, strict=False)

        if config.MODEL.PHYSFORMER.PRETRAINED_PATH:
            phys_state = torch.load(config.MODEL.PHYSFORMER.PRETRAINED_PATH)
            # Handle DataParallel prefix
            new_phys_state = {}
            for k, v in phys_state.items():
                if k.startswith('module.'):
                    new_phys_state[k[7:]] = v
                else:
                    new_phys_state[k] = v
            self.model.physformer.load_state_dict(new_phys_state, strict=False)
```

---

## 7. Training Loop

### 7.1 Standard Training Loop

```python
def train(self, data_loader):
    mean_training_losses = []
    mean_valid_losses = []

    for epoch in range(self.max_epoch_num):
        print(f"==== Training Epoch: {epoch} ====")
        self.model.train()

        tbar = tqdm(data_loader["train"], ncols=80)
        loss_rPPG_avg = []

        for idx, batch in enumerate(tbar):
            # Get data (format depends on loader)
            # For standard loaders: data, label, filename, chunk_id
            # For STVENLoader: compressed, uncompressed, bitrate_label, bvp_label
            video = batch[0].float().to(self.device)
            bvp_label = batch[3].float().to(self.device)

            # Forward pass
            rPPG_pred = self.model(video)

            # Normalize prediction and label
            rPPG_pred_norm = (rPPG_pred - rPPG_pred.mean(axis=-1, keepdim=True)) / \
                             rPPG_pred.std(axis=-1, keepdim=True)
            bvp_label_norm = (bvp_label - bvp_label.mean(axis=-1, keepdim=True)) / \
                             bvp_label.std(axis=-1, keepdim=True)

            # Calculate loss
            loss = self.criterion_Pearson(rPPG_pred_norm, bvp_label_norm)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_rPPG_avg.append(loss.item())
            tbar.set_description(f"Loss: {loss.item():.4f}")

        epoch_avg_loss = np.mean(loss_rPPG_avg)
        mean_training_losses.append(epoch_avg_loss)

        # Save checkpoint
        self.save_model(epoch)

        # Validation
        self.current_epoch = epoch
        valid_loss = self.valid(data_loader)
        if valid_loss is not None:
            mean_valid_losses.append(valid_loss)

    # Print best epoch
    print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")

    # Plot losses
    if self.config.TRAIN.PLOT_LOSSES_AND_LR:
        self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, [], self.config)
```

### 7.2 Joint Training Example (STVEN + PhysFormer)

```python
def train(self, data_loader):
    for epoch in range(self.max_epoch_num):
        self.model.train()
        self.model.physformer.eval()  # Frozen backend

        for batch in tqdm(data_loader["train"]):
            compressed_vid = batch[0].float().to(self.device)
            bvp_label = batch[3].float().to(self.device)

            # Ground truth normalization
            teacher_rPPG = (bvp_label - bvp_label.mean(axis=-1, keepdim=True)) / \
                           bvp_label.std(axis=-1, keepdim=True)

            # Create one-hot label for STVEN (always use HQ label=0)
            num_classes = self.model.stven.num_bitrate_levels
            student_label = torch.zeros(compressed_vid.shape[0], num_classes).to(self.device)
            student_label[:, 0] = 1.0

            # Forward: compressed -> STVEN -> PhysFormer -> rPPG
            student_rPPG, _, _, _ = self.model(compressed_vid, student_label, 2.0)

            # Normalize prediction
            student_rPPG_norm = (student_rPPG - student_rPPG.mean(axis=-1, keepdim=True)) / \
                                student_rPPG.std(axis=-1, keepdim=True)

            # Loss
            loss = self.criterion_Pearson(student_rPPG_norm, teacher_rPPG)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

---

## 8. Validation and Model Selection

### 8.1 Validation Loop

```python
def valid(self, data_loader):
    if data_loader["valid"] is None:
        print("No data for valid")
        return None

    self.model.eval()
    valid_loss = []

    with torch.no_grad():
        for batch in tqdm(data_loader["valid"]):
            video = batch[0].float().to(self.device)
            bvp_label = batch[3].float().to(self.device)

            # Normalize
            teacher_rPPG = (bvp_label - bvp_label.mean(axis=-1, keepdim=True)) / \
                           bvp_label.std(axis=-1, keepdim=True)

            # Forward
            rPPG_pred = self.model(video)
            rPPG_pred_norm = (rPPG_pred - rPPG_pred.mean(axis=-1, keepdim=True)) / \
                             rPPG_pred.std(axis=-1, keepdim=True)

            # Loss
            loss = self.criterion_Pearson(rPPG_pred_norm, teacher_rPPG)
            valid_loss.append(loss.item())

    avg_val_loss = np.mean(valid_loss)

    # Model selection
    if self.min_valid_loss is None:
        self.min_valid_loss = avg_val_loss
        self.best_epoch = self.current_epoch
    elif avg_val_loss < self.min_valid_loss:
        self.min_valid_loss = avg_val_loss
        self.best_epoch = self.current_epoch

    return avg_val_loss
```

### 8.2 Model Selection Logic

- Track `min_valid_loss` and `best_epoch`
- Save best model based on validation performance
- Controlled by `TEST.USE_LAST_EPOCH`:
  - `False`: Use best epoch model (recommended)
  - `True`: Use last epoch model

---

## 9. Testing and Evaluation

### 9.1 Test Loop

```python
def test(self, data_loader):
    # Load checkpoint
    if self.config.TOOLBOX_MODE == "only_test":
        self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
    else:
        if self.config.TEST.USE_LAST_EPOCH:
            model_path = os.path.join(self.model_dir,
                f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth")
        else:
            model_path = os.path.join(self.model_dir,
                f"{self.model_file_name}_Epoch{self.best_epoch}.pth")
        self.model.load_state_dict(torch.load(model_path))

    self.model.eval()
    predictions = dict()
    labels = dict()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader["test"])):
            video = batch[0].float().to(self.device)
            bvp_label = batch[3].float().to(self.device)
            subj_index = batch[2][0]  # or batch[2][i] for batch > 1
            sort_index = int(batch[3][0])

            # Inference
            rPPG_pred = self.model(video)

            # Store predictions
            if subj_index not in predictions:
                predictions[subj_index] = dict()
                labels[subj_index] = dict()

            predictions[subj_index][sort_index] = rPPG_pred[i]
            labels[subj_index][sort_index] = bvp_label[i]

    # Calculate metrics
    calculate_metrics(predictions, labels, self.config)

    # Save outputs (optional)
    if self.config.TEST.OUTPUT_SAVE_DIR:
        self.save_test_outputs(predictions, labels, self.config)
```

### 9.2 Post-Processing

The framework supports two evaluation methods (`INFERENCE.EVALUATION_METHOD`):

1. **FFT**: Frequency-domain heart rate estimation
2. **Peak Detection**: Time-domain peak detection

```python
# In metrics.py
if config.INFERENCE.EVALUATION_METHOD == "FFT":
    gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
        pred_window, label_window,
        diff_flag=diff_flag_test,
        fs=config.TEST.DATA.FS,
        hr_method='FFT'
    )
elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
    gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
        pred_window, label_window,
        diff_flag=diff_flag_test,
        fs=config.TEST.DATA.FS,
        hr_method='Peak'
    )
```

### 9.3 Evaluation Window

Configurable evaluation window size:
```yaml
INFERENCE:
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: True
    WINDOW_SIZE: 20  # seconds
```

---

## 10. Loss Functions

### 10.1 NegPearson Loss (Primary Loss)

```python
class Neg_Pearson(nn.Module):
    def forward(self, pred, target):
        """
        pred: [B, T]
        target: [B, T]
        """
        pred_mean = pred.mean(dim=-1, keepdim=True)
        target_mean = target.mean(dim=-1, keepdim=True)
        pred_c = pred - pred_mean
        target_c = target - target_mean

        numerator = (pred_c * target_c).sum(dim=-1)
        denominator = torch.sqrt(
            (pred_c ** 2).sum(dim=-1) *
            (target_c ** 2).sum(dim=-1) + 1e-8
        )

        return (1.0 - numerator / denominator).mean()
```

### 10.2 PhysFormer Loss Computer

```python
class TorchLossComputer:
    def __init__(self, loss_1='NegPearson', loss_2=None, loss_3=None,
                 weight_loss_1=1.0, weight_loss_2=0.0, weight_loss_3=0.0):
        self.loss_1 = loss_1
        self.weight_loss_1 = weight_loss_1
        # ... additional losses

    def get_loss(self, preds, bvps, gra_sharp=2.0):
        total_loss = 0
        # Calculate and combine losses
        return total_loss
```

---

## 11. Metrics

### 11.1 Supported Metrics

Configured via `TEST.METRICS`:

| Metric | Description |
|--------|-------------|
| `MAE` | Mean Absolute Error (bpm) |
| `RMSE` | Root Mean Square Error (bpm) |
| `MAPE` | Mean Absolute Percentage Error (%) |
| `Pearson` | Pearson correlation coefficient |
| `SNR` | Signal-to-Noise Ratio (dB) |
| `MACC` | Mean Absolute Correlation Coefficient |
| `BA` | Bland-Altman analysis (generates plots) |

### 11.2 Metric Calculation

```python
def calculate_metrics(predictions, labels, config):
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()

    for index in predictions.keys():
        pred = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        # Process in windows
        for i in range(0, len(pred), window_frame_size):
            pred_window = pred[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window,
                    diff_flag=diff_flag_test,
                    fs=config.TEST.DATA.FS,
                    hr_method='FFT'
                )
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)

    # Calculate aggregate metrics
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)

    for metric in config.TEST.METRICS:
        if metric == "MAE":
            MAE = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
            standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / \
                             np.sqrt(num_test_samples)
            print(f"MAE: {MAE} +/- {standard_error}")
        elif metric == "Pearson":
            Pearson = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
            correlation_coefficient = Pearson[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
            print(f"Pearson: {correlation_coefficient} +/- {standard_error}")
        elif metric == "BA":
            # Generate Bland-Altman plots
            compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
            compare.scatter_plot(...)
            compare.difference_plot(...)
```

---

## 12. Integration Guide for New Architectures

### 12.1 Step-by-Step Integration

#### Step 1: Create Model File

Create `neural_methods/model/QAFCPhysFormer.py`:

```python
import torch
import torch.nn as nn

class QAFCPhysFormer(nn.Module):
    """Quality-Aware Feature Conditioning PhysFormer"""

    def __init__(self, physformer_config, quality_dim=16):
        super().__init__()

        # Quality branch
        self.quality_spatial = QualitySpatialEncoder(out_channels=64)
        self.quality_temporal = QualityTemporalEncoder(
            input_dim=64, hidden_dim=64, quality_dim=quality_dim
        )
        self.quality_scalar = QualityScalarHead(quality_dim)

        # PhysFormer backbone with FiLM
        self.backbone = QualityAwarePhysFormer(
            physformer_config, quality_dim=quality_dim
        )

    def forward(self, video, return_quality=False):
        # Quality branch
        q_spatial = self.quality_spatial(video)
        quality_emb = self.quality_temporal(q_spatial)

        # rPPG prediction with quality conditioning
        rppg = self.backbone(video, quality_emb)

        if return_quality:
            quality_score = self.quality_scalar(quality_emb)
            return rppg, quality_emb, quality_score
        return rppg
```

#### Step 2: Create Trainer File

Create `neural_methods/trainer/QAFCPhysFormerTrainer.py`:

```python
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.QAFCPhysFormer import QAFCPhysFormer
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from evaluation.metrics import calculate_metrics

class QAFCPhysFormerTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.config = config

        # Initialize model
        self.model = QAFCPhysFormer(
            physformer_config=config.MODEL.PHYSFORMER,
            quality_dim=config.MODEL.QAFC_PHYSFORMER.QUALITY_DIM
        ).to(self.device)

        # Load pretrained PhysFormer weights (if specified)
        self._load_pretrained_weights(config)

        # Freeze PhysFormer backbone (optional)
        if config.MODEL.QAFC_PHYSFORMER.FREEZE_BACKBONE:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.TRAIN.LR
        )

        # Loss functions
        self.criterion = TorchLossComputer(
            loss_1='NegPearson',
            weight_loss_1=1.0
        )

        # Quality ranking loss (for self-supervised quality learning)
        self.ranking_loss = QualityRankingLoss(margin=0.1)

        # Tracking
        self.min_valid_loss = None
        self.best_epoch = 0

    def _load_pretrained_weights(self, config):
        if config.MODEL.PHYSFORMER.PRETRAINED_PATH:
            phys_state = torch.load(config.MODEL.PHYSFORMER.PRETRAINED_PATH)
            # Handle DataParallel prefix
            new_phys_state = {}
            for k, v in phys_state.items():
                if k.startswith('module.'):
                    new_phys_state[k[7:]] = v
                else:
                    new_phys_state[k] = v
            self.model.backbone.load_state_dict(new_phys_state, strict=False)

    def train(self, data_loader):
        mean_training_losses = []
        mean_valid_losses = []

        for epoch in range(self.max_epoch_num):
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            epoch_losses = []

            for batch in tbar:
                video = batch[0].float().to(self.device)
                bvp_label = batch[3].float().to(self.device)

                # Forward
                pred_rppg, quality_emb, quality_score = self.model(
                    video, return_quality=True
                )

                # Calculate rPPG loss
                loss = self.criterion.get_loss(pred_rppg, bvp_label)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                tbar.set_description(f"Loss: {loss.item():.4f}")

            epoch_avg_loss = np.mean(epoch_losses)
            mean_training_losses.append(epoch_avg_loss)

            self.save_model(epoch)

            # Validation
            self.current_epoch = epoch
            valid_loss = self.valid(data_loader)
            if valid_loss is not None:
                mean_valid_losses.append(valid_loss)

        # Save best epoch info
        print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")

        # Plot losses
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, [], self.config)

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            return None

        self.model.eval()
        valid_loss = []

        with torch.no_grad():
            for batch in tqdm(data_loader["valid"]):
                video = batch[0].float().to(self.device)
                bvp_label = batch[3].float().to(self.device)

                pred_rppg = self.model(video)
                loss = self.criterion.get_loss(pred_rppg, bvp_label)
                valid_loss.append(loss.item())

        avg_val_loss = np.mean(valid_loss)

        # Model selection
        if self.min_valid_loss is None or avg_val_loss < self.min_valid_loss:
            self.min_valid_loss = avg_val_loss
            self.best_epoch = self.current_epoch

        return avg_val_loss

    def test(self, data_loader):
        # Load checkpoint
        if self.config.TOOLBOX_MODE == "only_test":
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(self.model_dir,
                    f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth")
            else:
                model_path = os.path.join(self.model_dir,
                    f"{self.model_file_name}_Epoch{self.best_epoch}.pth")
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        predictions = dict()
        labels = dict()

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(data_loader["test"])):
                video = batch[0].float().to(self.device)
                bvp_label = batch[3].float().to(self.device)
                subj_index = batch[2][0]
                sort_index = int(batch[3][0])

                pred_rppg = self.model(video)

                if subj_index not in predictions:
                    predictions[subj_index] = dict()
                    labels[subj_index] = dict()

                predictions[subj_index][sort_index] = pred_rppg[0]
                labels[subj_index][sort_index] = bvp_label[0]

        calculate_metrics(predictions, labels, self.config)

        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir,
            f"{self.model_file_name}_Epoch{index}.pth")
        torch.save(self.model.state_dict(), model_path)
```

#### Step 3: Add Model Config

Add to `config.py`:

```python
# -----------------------------------------------------------------------------
# Model Settings for QAFC-PhysFormer
# -----------------------------------------------------------------------------
_C.MODEL.QAFC_PHYSFORMER = CN()
_C.MODEL.QAFC_PHYSFORMER.QUALITY_DIM = 16
_C.MODEL.QAFC_PHYSFORMER.FREEZE_BACKBONE = True
_C.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH = ''
```

#### Step 4: Register Model in main.py

Add to `train_and_test()` and `test()` functions:

```python
elif config.MODEL.NAME == 'QAFCPhysFormer':
    model_trainer = trainer.QAFCPhysFormerTrainer.QAFCPhysFormerTrainer(config, data_loader_dict)
```

#### Step 5: Create Config File

Create `configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml`:

```yaml
BASE: [""]
TOOLBOX_MODE: "train_and_test"
TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 50
  LR: 1e-4
  MODEL_FILE_NAME: "QAFCPhysFormer"
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_PATH: "/path/to/UBFC-rPPG"
    CACHED_PATH: "/path/to/UBFC-rPPG-cache"
    BEGIN: 0.0
    END: 0.8
    PREPROCESS:
      DATA_TYPE: ["DiffNormalized"]
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 160
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: "HC"
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
      RESIZE:
        H: 128
        W: 128

VALID:
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_PATH: "/path/to/UBFC-rPPG"
    CACHED_PATH: "/path/to/UBFC-rPPG-cache"
    BEGIN: 0.8
    END: 0.9

TEST:
  METRICS: ["MAE", "RMSE", "MAPE", "Pearson", "BA"]
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: PURE
    DO_PREPROCESS: False
    DATA_PATH: "/path/to/PURE"
    CACHED_PATH: "/path/to/PURE-cache"
    BEGIN: 0.9
    END: 1.0

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

MODEL:
  DROP_RATE: 0.2
  NAME: QAFCPhysFormer
  MODEL_DIR: "PreTrainedModels"
  PHYSFORMER:
    PRETRAINED_PATH: "./final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth"
    PATCH_SIZE: 4
    DIM: 96
    FF_DIM: 144
    NUM_HEADS: 4
    NUM_LAYERS: 12
    THETA: 0.7
    IMAGE_SIZE: [160, 128, 128]
    PATCHES: [4, 4, 4]
    DROPOUT_RATE: 0.2
  QAFC_PHYSFORMER:
    QUALITY_DIM: 16
    FREEZE_BACKBONE: True

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: True
    WINDOW_SIZE: 20
```

#### Step 6: Run Training

```bash
python main.py --config_file ./configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml
```

### 12.2 Special Considerations for QAFC-PhysFormer

#### Multi-CRF Data Handling

If using multi-CRF training (for quality ranking loss), you may need to:

1. **Use STVENLoader or create a custom loader** that returns multiple CRF versions
2. **Modify the training loop** to handle quality ranking pairs:

```python
def train(self, data_loader):
    for batch in data_loader["train"]:
        # If using multi-CRF setup
        video_high = batch['video_high'].to(self.device)  # Low CRF (high quality)
        video_low = batch['video_low'].to(self.device)    # High CRF (low quality)
        bvp_label = batch['rppg'].to(self.device)

        # Forward both through quality branch
        _, quality_emb_high, quality_score_high = self.model(video_high, return_quality=True)
        _, quality_emb_low, quality_score_low = self.model(video_low, return_quality=True)

        # rPPG loss (use high quality)
        pred_rppg = self.model(video_high)
        loss_rppg = self.criterion.get_loss(pred_rppg, bvp_label)

        # Quality ranking loss
        loss_ranking = self.ranking_loss(quality_score_high, quality_score_low)

        # Combined loss
        total_loss = loss_rppg + 0.1 * loss_ranking
```

#### Three-Phase Training

Implement phased training in the trainer:

```python
def train(self, data_loader):
    phase_boundaries = {1: 10, 2: 70, 3: 100}

    for epoch in range(self.max_epoch_num):
        # Determine phase
        if epoch < phase_boundaries[1]:
            phase = 1
            # Freeze backbone, train quality branch only
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        elif epoch < phase_boundaries[2]:
            phase = 2
            # Unfreeze all for joint training
            for param in self.model.backbone.parameters():
                param.requires_grad = True
        else:
            phase = 3
            # Freeze quality branch, fine-tune backbone
            for param in self.model.quality_spatial.parameters():
                param.requires_grad = False

        # Create new optimizer for phase
        optimizer = self._get_param_groups(phase)

        # ... rest of training loop
```

---

## Appendix A: File Paths Summary

| Component | Path |
|-----------|------|
| Training configs | `configs/train_configs/` |
| Inference configs | `configs/infer_configs/` |
| Models | `neural_methods/model/` |
| Trainers | `neural_methods/trainer/` |
| Losses | `neural_methods/loss/` |
| Data loaders | `dataset/data_loader/` |
| Metrics | `evaluation/metrics.py` |
| Unsupervised methods | `unsupervised_methods/methods/` |

## Appendix B: Common Commands

```bash
# Train and test
python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml

# Test only
python main.py --config_file ./configs/infer_configs/PURE_UBFC-rPPG_TSCAN_BASIC.yaml

# Unsupervised methods
python main.py --config_file ./configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml

# Clear preprocessing cache
rm -rf /path/to/dataset-cache/*
```

## Appendix C: Key Design Patterns

1. **Factory Pattern**: Model instantiation based on config.MODEL.NAME
2. **Strategy Pattern**: Different data loaders for different datasets
3. **Template Method Pattern**: BaseTrainer defines skeleton, concrete trainers implement details
4. **Composite Pattern**: PhysFormerWithSTVEN combines multiple models
5. **Configuration-Driven**: All hyperparameters via YAML configs

---

**End of Document**
