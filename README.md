# HAR-FPGA: Human Activity Recognition for FPGA Deployment

Multi-architecture training pipeline for Human Activity Recognition (HAR) using the
UCI HAR Smartphones dataset. Supports three model architectures for comparison,
with FPGA export (JSON spec + `.mem` weight files) and post-training quantization.

Based on reference [10] from the research paper on efficient FPGA implementation of
neural networks for HAR (DCLSTM achieving 97.8% and WCLSTM achieving 98.9% on UCI HAR).

---

## Table of Contents

- [Supported Models](#supported-models)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [CLI Reference](#cli-reference)
  - [Training](#training)
  - [Inference / Test Evaluation](#inference--test-evaluation)
  - [Export for FPGA](#export-for-fpga)
  - [Quantization](#quantization)
  - [Model Comparison](#model-comparison)
- [Model Architectures](#model-architectures)
  - [1D-CNN (1dcnn)](#1d-cnn-1dcnn)
  - [CNN+LSTM / DCLSTM (cnn_lstm)](#cnnlstm--dclstm-cnn_lstm)
  - [Wavelet CNN+LSTM / WCLSTM (wclstm)](#wavelet-cnnlstm--wclstm-wclstm)
- [Dataset](#dataset)
  - [Data Modes](#data-modes)
  - [Label Mapping](#label-mapping)
- [Preprocessing](#preprocessing)
- [Artifacts Reference (for Hardware Team)](#artifacts-reference-for-hardware-team)
- [Source Code Reference](#source-code-reference)
- [FPGA Implementation Notes](#fpga-implementation-notes)
- [Training Configuration](#training-configuration)
- [Results](#results)

---

## Supported Models

| Model ID   | Architecture           | Input Data             | Parameters | Reference        |
|------------|------------------------|------------------------|------------|------------------|
| `1dcnn`    | 1D-CNN                 | 19 statistical features| ~305       | Baseline         |
| `cnn_lstm` | CNN + LSTM (DCLSTM)    | 128x9 raw inertial     | ~122,949   | Paper ref [10]   |
| `wclstm`   | Wavelet CNN+LSTM       | Wavelet-transformed    | ~122,949   | Paper ref [10]   |

All three models use the same `--model` flag across all CLI commands.

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Train all three models
uv run python -m har_fpga.train --model 1dcnn
uv run python -m har_fpga.train --model cnn_lstm
uv run python -m har_fpga.train --model wclstm

# 3. Compare results
uv run python -m har_fpga.compare --plot

# 4. Quantize any model
uv run python -m har_fpga.quantize --model 1dcnn
uv run python -m har_fpga.quantize --model cnn_lstm
uv run python -m har_fpga.quantize --model wclstm

# 5. Export any model for FPGA
uv run python -m har_fpga.export --model 1dcnn
```

---

## Project Structure

```
har-fpga/
├── pyproject.toml                  # Project config, dependencies, entry points
├── README.md                       # This file
│
├── configs/
│   ├── features.json               # 19 selected feature names + indices (for 1dcnn)
│   └── training.json               # Hyperparameters, label map, per-model configs
│
├── src/har_fpga/                   # Source code
│   ├── __init__.py
│   ├── data.py                     # Load UCI HAR: features mode (19) or raw mode (128x9)
│   ├── preprocess.py               # Z-score normalization (fit / transform / save / load)
│   ├── model.py                    # Build all 3 model architectures + extract spec
│   ├── train.py                    # Training pipeline with --model flag
│   ├── export.py                   # Export weights to .mem + JSON spec
│   ├── infer.py                    # Run inference / test evaluation
│   ├── quantize.py                 # Post-training quantization comparison
│   └── compare.py                  # Cross-model comparison tables + plots
│
├── artifacts/                      # Generated after training (per-model subdirectories)
│   ├── 1dcnn/                      # Artifacts for the 1D-CNN model
│   │   ├── har_model.keras
│   │   ├── model_spec.json
│   │   ├── model_weights.mem
│   │   ├── weights_readable.txt
│   │   ├── scaler.json
│   │   ├── training_history.json
│   │   └── quantization/           # Quantization results (after running quantize)
│   │       ├── results.json
│   │       ├── quantization_results.png
│   │       ├── fp16/
│   │       ├── int16/
│   │       └── int8/
│   ├── cnn_lstm/                   # Artifacts for the CNN+LSTM model (same structure)
│   ├── wclstm/                     # Artifacts for the WCLSTM model (same structure)
│   └── comparison.png              # Cross-model comparison plot (after running compare)
│
└── data/                           # Downloaded dataset (auto-cached, not committed)
    ├── UCI_HAR_Dataset.zip
    └── UCI HAR Dataset/
```

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

Install `uv` if you don't have it:
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

Clone the repo and install dependencies:

```bash
cd har-fpga
uv sync
```

This creates a `.venv`, resolves all dependencies (TensorFlow, NumPy, scikit-learn,
requests, matplotlib, PyWavelets), and installs the project in editable mode.

---

## CLI Reference

Every command accepts `--model {1dcnn,cnn_lstm,wclstm}` to select the architecture.
The default is `1dcnn` if `--model` is omitted.

### Training

```bash
# Train the 1D-CNN (default, 19 features, ~305 params)
uv run python -m har_fpga.train --model 1dcnn

# Train the CNN+LSTM (DCLSTM, raw 128x9 signals, ~123K params)
uv run python -m har_fpga.train --model cnn_lstm

# Train the WCLSTM (wavelet-transformed signals, ~123K params)
uv run python -m har_fpga.train --model wclstm

# Custom hyperparameters (works with any model)
uv run python -m har_fpga.train --model cnn_lstm --epochs 50 --batch-size 32 --lr 0.0005

# Force CPU training (skip GPU detection)
uv run python -m har_fpga.train --model cnn_lstm --no-gpu
```

**Training arguments:**

| Argument       | Description                              | Default                   |
|----------------|------------------------------------------|---------------------------|
| `--model`      | Model architecture                       | `1dcnn`                   |
| `--epochs`     | Number of training epochs                | 30 (from training.json)   |
| `--batch-size` | Mini-batch size                          | 64 (from training.json)   |
| `--lr`         | Learning rate                            | 0.001 (from training.json)|
| `--no-gpu`     | Force CPU-only training                  | False                     |

**Output:** All artifacts saved to `artifacts/<model_type>/`.

On first run, the UCI HAR dataset (~61 MB) is automatically downloaded and cached in `data/`.

### Inference / Test Evaluation

```bash
# Evaluate 1D-CNN on the UCI HAR test split
uv run python -m har_fpga.infer --model 1dcnn --test

# Evaluate CNN+LSTM on the test split
uv run python -m har_fpga.infer --model cnn_lstm --test

# Evaluate WCLSTM on the test split
uv run python -m har_fpga.infer --model wclstm --test

# Single sample prediction (1D-CNN only, 19 comma-separated features)
uv run python -m har_fpga.infer --model 1dcnn --sample "0.289,-0.020,-0.133,-0.995,-0.983,-0.914,-0.727,0.117,0.679,-0.998,-0.975,-0.960,-0.112,-0.065,0.029,-0.032,-0.563,-0.440,-0.202"

# Batch inference from file (1D-CNN only, one 19-value sample per line)
uv run python -m har_fpga.infer --model 1dcnn --file path/to/samples.txt
```

**Inference arguments:**

| Argument     | Description                                    | Notes                    |
|--------------|------------------------------------------------|--------------------------|
| `--model`    | Model architecture                             | `1dcnn` / `cnn_lstm` / `wclstm` |
| `--test`     | Evaluate on UCI HAR test split                 | Works with all models    |
| `--sample`   | Single 19-value comma-separated prediction     | 1dcnn only               |
| `--file`     | Batch prediction from text file                | 1dcnn only               |

**Note:** `--sample` and `--file` are only supported for `1dcnn` because the other models
require raw inertial signals (128 timesteps x 9 channels) as input.

### Export for FPGA

```bash
# Export 1D-CNN model weights and spec for FPGA
uv run python -m har_fpga.export --model 1dcnn

# Export CNN+LSTM
uv run python -m har_fpga.export --model cnn_lstm

# Export WCLSTM
uv run python -m har_fpga.export --model wclstm

# Export from a specific model file
uv run python -m har_fpga.export --model 1dcnn --model-path path/to/model.keras
```

**Export arguments:**

| Argument       | Description                              | Default                              |
|----------------|------------------------------------------|--------------------------------------|
| `--model`      | Model architecture                       | `1dcnn`                              |
| `--model-path` | Override Keras model file path           | `artifacts/<model>/har_model.keras`  |

**Output files (in `artifacts/<model_type>/`):**
- `model_spec.json` -- Architecture for the hardware team
- `model_weights.mem` -- Weights in IEEE-754 hex for `$readmemh`
- `weights_readable.txt` -- Human-readable weight values

### Quantization

```bash
# Quantize the 1D-CNN model (FP32 vs FP16 vs INT16 vs INT8)
uv run python -m har_fpga.quantize --model 1dcnn

# Quantize the CNN+LSTM model
uv run python -m har_fpga.quantize --model cnn_lstm

# Quantize the WCLSTM model
uv run python -m har_fpga.quantize --model wclstm

# Quantize from a specific model file
uv run python -m har_fpga.quantize --model cnn_lstm --model-path path/to/model.keras
```

**Quantization arguments:**

| Argument       | Description                              | Default                              |
|----------------|------------------------------------------|--------------------------------------|
| `--model`      | Model architecture                       | `1dcnn`                              |
| `--model-path` | Override Keras model file path           | `artifacts/<model>/har_model.keras`  |

**Output (in `artifacts/<model_type>/quantization/`):**
- `results.json` -- Accuracy, inference time, weight sizes per variant
- `quantization_results.png` -- Comparison bar charts
- `fp16/` -- FP16 weights (`.mem` + `metadata.json`)
- `int16/` -- INT16 weights (`.mem` + `metadata.json`)
- `int8/` -- INT8 weights (`.mem` + `metadata.json`)

### Model Comparison

```bash
# Print comparison table of all trained models
uv run python -m har_fpga.compare

# Print comparison table + generate comparison bar chart
uv run python -m har_fpga.compare --plot
```

**Compare arguments:**

| Argument  | Description                                    |
|-----------|------------------------------------------------|
| `--plot`  | Generate comparison bar chart (artifacts/comparison.png) |

**Output:** Prints a table comparing test accuracy, parameter count, and training time
across all trained models. If quantization has been run, also shows FP32 vs INT8
accuracy comparison. With `--plot`, saves `artifacts/comparison.png`.

---

## Model Architectures

### 1D-CNN (1dcnn)

A minimal 1D-CNN with **305 total parameters** (1.19 KB), designed for direct FPGA translation.
Operates on 19 pre-extracted statistical features from the UCI HAR dataset.

```
Input (19, 1)
  |
  v
Conv1D(filters=12, kernel_size=19, stride=1, padding='valid', activation='relu')
  |  -- output shape: (1, 12)
  v
Flatten
  |  -- output shape: (12,)
  v
Dense(units=5, activation='softmax')
  |
  v
Output (5,)  -->  [WALKING, SITTING, STANDING, LAYING, TRANSITION]
```

| Layer        | Type    | Output Shape | Parameters | Activation |
|--------------|---------|-------------|------------|------------|
| input        | Input   | (None,19,1) | 0          | --         |
| conv1d       | Conv1D  | (None,1,12) | 240        | ReLU       |
| flatten      | Flatten | (None,12)   | 0          | --         |
| dense_output | Dense   | (None,5)    | 65         | Softmax    |
| **Total**    |         |             | **305**    |            |

### CNN+LSTM / DCLSTM (cnn_lstm)

Combines CNNs with LSTM networks to leverage spatial and temporal feature extraction.
Operates on raw inertial signals (128 timesteps x 9 channels).

```
Input (128, 9)
  |
  v
Conv1D(64, kernel_size=5, padding='same', activation='relu')
  |  -- output shape: (128, 64)
  v
Conv1D(64, kernel_size=5, padding='same', activation='relu')
  |  -- output shape: (128, 64)
  v
Dropout(0.3)
  |
  v
LSTM(128)
  |  -- output shape: (128,)
  v
Dropout(0.3)
  |
  v
Dense(5, activation='softmax')
  |
  v
Output (5,)  -->  [WALKING, SITTING, STANDING, LAYING, TRANSITION]
```

| Layer        | Type    | Output Shape   | Parameters | Activation        |
|--------------|---------|---------------|------------|-------------------|
| input        | Input   | (None,128,9)  | 0          | --                |
| conv1d_1     | Conv1D  | (None,128,64) | 2,944      | ReLU              |
| conv1d_2     | Conv1D  | (None,128,64) | 20,544     | ReLU              |
| dropout_1    | Dropout | (None,128,64) | 0          | --                |
| lstm         | LSTM    | (None,128)    | 98,816     | tanh/sigmoid      |
| dropout_2    | Dropout | (None,128)    | 0          | --                |
| dense_output | Dense   | (None,5)      | 645        | Softmax           |
| **Total**    |         |               | **122,949**|                   |

### Wavelet CNN+LSTM / WCLSTM (wclstm)

Incorporates wavelet transforms (WTs) to enhance feature extraction by providing
time-frequency analysis. The raw signals are first decomposed using a Daubechies-4
wavelet at level 2 before being fed into the CNN+LSTM network.

```
Raw Signal (128, 9)
  |
  v  [Wavelet Decomposition: db4, level=2]
  |  Concatenates approximation + detail coefficients
  v
Input (141, 9)   <-- 141 = concat of wavelet coefficient lengths
  |
  v
Conv1D(64, kernel_size=5, padding='same', activation='relu')
  |  -- output shape: (141, 64)
  v
Conv1D(64, kernel_size=5, padding='same', activation='relu')
  |  -- output shape: (141, 64)
  v
Dropout(0.3)
  |
  v
LSTM(128)
  |  -- output shape: (128,)
  v
Dropout(0.3)
  |
  v
Dense(5, activation='softmax')
  |
  v
Output (5,)
```

**Wavelet configuration:**
- Wavelet: Daubechies-4 (`db4`)
- Decomposition level: 2
- Output: Concatenation of `[cA2, cD2, cD1]` coefficients per channel
- 128 timesteps -> 141 wavelet coefficients per channel

---

## Dataset

**UCI Human Activity Recognition Using Smartphones**

- Source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- 10,299 samples total (7,352 train + 2,947 test)
- Raw data: accelerometer + gyroscope from smartphone worn at the waist

### Data Modes

| Model    | Data Mode   | Input Shape   | Description                              |
|----------|-------------|---------------|------------------------------------------|
| `1dcnn`  | `features`  | (N, 19)       | 19 selected statistical features         |
| `cnn_lstm`| `raw`      | (N, 128, 9)   | 9 raw inertial signal channels           |
| `wclstm` | `wavelet`   | (N, 141, 9)   | Wavelet-decomposed inertial signals      |

**19 statistical features (for 1dcnn):**

| #  | Feature Name             | UCI Index | Category                        |
|----|--------------------------|----------:|---------------------------------|
| 1  | tBodyAcc-mean()-X        | 1         | Body acceleration mean          |
| 2  | tBodyAcc-mean()-Y        | 2         | Body acceleration mean          |
| 3  | tBodyAcc-mean()-Z        | 3         | Body acceleration mean          |
| 4  | tBodyAcc-std()-X         | 4         | Body acceleration std           |
| 5  | tBodyAcc-std()-Y         | 5         | Body acceleration std           |
| 6  | tBodyAcc-std()-Z         | 6         | Body acceleration std           |
| 7  | tGravityAcc-mean()-X     | 41        | Gravity acceleration mean       |
| 8  | tGravityAcc-mean()-Y     | 42        | Gravity acceleration mean       |
| 9  | tGravityAcc-mean()-Z     | 43        | Gravity acceleration mean       |
| 10 | tGravityAcc-std()-X      | 44        | Gravity acceleration std        |
| 11 | tGravityAcc-std()-Y      | 45        | Gravity acceleration std        |
| 12 | tGravityAcc-std()-Z      | 46        | Gravity acceleration std        |
| 13 | tBodyAccMag-mean()       | 201       | Body acceleration magnitude     |
| 14 | tBodyAccMag-std()        | 202       | Body acceleration magnitude     |
| 15 | tGravityAccMag-mean()    | 214       | Gravity acceleration magnitude  |
| 16 | tGravityAccMag-std()     | 215       | Gravity acceleration magnitude  |
| 17 | tBodyAccJerkMag-mean()   | 227       | Body jerk magnitude             |
| 18 | tBodyAccJerkMag-std()    | 228       | Body jerk magnitude             |
| 19 | tBodyGyroMag-mean()      | 253       | Gyroscope magnitude             |

**9 raw inertial signal channels (for cnn_lstm and wclstm):**

| #  | Signal Name    | Description                     |
|----|----------------|---------------------------------|
| 1  | body_acc_x     | Body acceleration X-axis        |
| 2  | body_acc_y     | Body acceleration Y-axis        |
| 3  | body_acc_z     | Body acceleration Z-axis        |
| 4  | body_gyro_x    | Body gyroscope X-axis           |
| 5  | body_gyro_y    | Body gyroscope Y-axis           |
| 6  | body_gyro_z    | Body gyroscope Z-axis           |
| 7  | total_acc_x    | Total acceleration X-axis       |
| 8  | total_acc_y    | Total acceleration Y-axis       |
| 9  | total_acc_z    | Total acceleration Z-axis       |

### Label Mapping

Original UCI labels are merged into 5 classes:

| Original Label (UCI) | Original ID | Mapped Class | Mapped ID |
|----------------------|-------------|-------------|-----------|
| WALKING              | 1           | WALKING     | 0         |
| WALKING_UPSTAIRS     | 2           | WALKING     | 0         |
| WALKING_DOWNSTAIRS   | 3           | WALKING     | 0         |
| SITTING              | 4           | SITTING     | 1         |
| STANDING             | 5           | STANDING    | 2         |
| LAYING               | 6           | LAYING      | 3         |
| Any other            | --          | TRANSITION  | 4         |

---

## Preprocessing

**Z-score normalization** is applied to all input data:

```
x_normalized = (x_raw - mean) / std
```

- `mean` and `std` are computed from the training set only.
- Saved to `artifacts/<model>/scaler.json`.
- **The FPGA must apply this same normalization before inference.**

For the 1D-CNN:
- Applied per feature (19 mean + 19 std values).
- After normalization, reshaped from `(19,)` to `(19, 1)`.

For CNN+LSTM / WCLSTM:
- Applied across all `timesteps x channels` (flattened, then reshaped back).
- For WCLSTM, wavelet decomposition is applied before normalization.

---

## Artifacts Reference (for Hardware Team)

After training, each model's artifacts are in `artifacts/<model_type>/`.

### model_spec.json

Layer-by-layer architecture specification including shapes, parameter counts,
activation functions, and kernel configurations.

### model_weights.mem

All weights as IEEE-754 float32 hex values, one per line.
Sections separated by `//` comment lines.
Compatible with Verilog `$readmemh`.

### scaler.json

Z-score normalization constants (mean + std per feature/channel).

### har_model.keras

Full saved Keras model for reloading in Python.

### weights_readable.txt

Human-readable weight dump with min/max/mean/std summaries.

### training_history.json

Per-epoch training metrics (loss, accuracy, val_loss, val_accuracy).

### quantization/ (after running quantize)

Post-training quantization results:
- `results.json` -- Per-variant accuracy and timing
- `quantization_results.png` -- Visual comparison
- `fp16/`, `int16/`, `int8/` -- Quantized weight files

---

## Source Code Reference

| File | Responsibility | Key Functions |
|------|---------------|---------------|
| `data.py` | Download UCI HAR, load features or raw signals, remap labels | `load_har_data()`, `load_har_raw()`, `remap_labels()` |
| `preprocess.py` | Z-score normalization | `ZScoreScaler.fit()`, `.transform()`, `.save()`, `.load()` |
| `model.py` | Build all 3 model architectures, extract spec | `build_model()`, `build_1dcnn()`, `build_cnn_lstm()`, `build_wclstm()`, `extract_model_spec()` |
| `train.py` | Full training pipeline with `--model` flag | `train()`, `main()` |
| `export.py` | Export weights to .mem + JSON spec | `export_weights_mem()`, `export_spec_json()` |
| `infer.py` | Run inference (single/batch/test) | `predict_single()`, `predict_batch()`, `main()` |
| `quantize.py` | Post-training quantization comparison | `quantize_symmetric()`, `main()` |
| `compare.py` | Cross-model comparison | `main()` |

---

## FPGA Implementation Notes

### 1D-CNN (305 params, 1.19 KB)

- **Compute:** 228 MACs (Conv1D) + 60 MACs (Dense) = 288 MACs total
- **Memory:** 305 x 32 bits = 1,220 bytes
- **Activations:** ReLU (sign bit check) + Softmax (or argmax)
- Effectively two fully-connected layers (kernel_size == input_length)

### CNN+LSTM (~123K params, ~480 KB)

- **Compute:** Significantly more than 1D-CNN due to LSTM recurrence
- **Conv layers:** Two Conv1D layers with 64 filters each
- **LSTM:** 128 hidden units, processes 128 timesteps sequentially
- **Memory:** ~480 KB for FP32 weights (INT8 reduces to ~120 KB)
- **Note:** LSTM requires sequential processing of timesteps, which may
  need pipelining or time-multiplexed computation on FPGA

### WCLSTM (~123K params, ~480 KB)

- Same network architecture as CNN+LSTM
- Requires wavelet decomposition preprocessing on FPGA
- Wavelet: db4 (Daubechies-4), level 2
- May achieve higher accuracy with same compute budget

---

## Training Configuration

All hyperparameters are in `configs/training.json`:

| Parameter        | Value                          | Applies To  |
|------------------|--------------------------------|-------------|
| Epochs           | 30                             | All models  |
| Batch Size       | 64                             | All models  |
| Learning Rate    | 0.001                          | All models  |
| Optimizer        | Adam                           | All models  |
| Loss Function    | Sparse Categorical Crossentropy| All models  |
| Validation Split | 20%                            | All models  |
| Random Seed      | 42                             | All models  |

Per-model config (conv filters, kernel sizes, LSTM units, dropout, wavelet params)
is in `configs/training.json` under `models.<model_type>`.

---

## Results

### Test Accuracy (UCI HAR Test Set, 2,947 samples)

| Model      | Test Accuracy | Parameters | FP32 Weight Size | Input Shape |
|------------|--------------|------------|------------------|-------------|
| **1D-CNN** | **92.67%**   | 305        | 1,220 B          | (19, 1)     |
| **CNN+LSTM** | **93.59%** | 122,949    | 491,796 B        | (128, 9)    |
| **WCLSTM** | **83.00%**   | 122,949    | 491,796 B        | (141, 9)    |

### Per-Class Accuracy

| Class     | 1D-CNN  | CNN+LSTM | WCLSTM  |
|-----------|---------|----------|---------|
| WALKING   | 100.00% | 99.86%   | 100.00% |
| SITTING   | 75.56%  | 86.35%   | 90.43%  |
| STANDING  | 87.03%  | 77.44%   | 14.85%  |
| LAYING    | 94.97%  | 100.00%  | 99.81%  |

### Quantization Results (FP32 vs INT8)

All models maintain accuracy through FP16 and INT16 quantization with zero degradation.
INT8 results:

| Model      | FP32 Acc | INT8 Acc | Acc Drop | FP32 Size  | INT8 Size  | Compression |
|------------|----------|----------|----------|------------|------------|-------------|
| **1D-CNN** | 92.67%   | 92.67%   | 0.00%    | 1,220 B    | 305 B      | 4x          |
| **CNN+LSTM** | 93.59% | 93.59%   | 0.00%    | 491,796 B  | 122,949 B  | 4x          |
| **WCLSTM** | 83.00%   | 82.69%   | 0.31%    | 491,796 B  | 122,949 B  | 4x          |

### Notes

- **CNN+LSTM achieves the best accuracy (93.59%)** among the three models on the UCI HAR test set with 5-class labels (WALKING, SITTING, STANDING, LAYING, TRANSITION).
- **1D-CNN is remarkably efficient** at 92.67% with only 305 parameters (1.19 KB), making it ideal for resource-constrained FPGA deployment.
- **WCLSTM underperforms expectations** at 83.00% vs the paper's reported 98.9%. The gap is likely due to: (a) our 5-class label merge vs the paper's original 6-class setup, (b) the 30-epoch training budget, and (c) possible differences in wavelet preprocessing. Additional tuning (more epochs, learning rate scheduling) may improve this.
- All three models show **excellent quantization robustness** -- INT8 quantization preserves accuracy with 4x weight compression, which is important for FPGA deployment with limited on-chip memory.

---

## GPU Support

TensorFlow on native Windows dropped GPU support after v2.10. The installed `tensorflow==2.18`
runs on CPU only on Windows.

- **1D-CNN:** ~305 params, trains in ~10s on CPU. GPU not needed.
- **CNN+LSTM / WCLSTM:** ~123K params, trains in ~2-5 minutes on CPU.
  For faster training, use WSL2 with `tensorflow[and-cuda]`.
