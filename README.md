# HAR-FPGA: Human Activity Recognition for FPGA Deployment

Lightweight 1D-CNN training pipeline for Human Activity Recognition (HAR) using the
UCI HAR Smartphones dataset. The trained model is exported to JSON specs and `.mem`
weight files for direct FPGA implementation.

Based on the research paper on efficient FPGA implementation of neural networks for HAR.

---

## Table of Contents

- [Results](#results)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [How to Run](#how-to-run)
  - [Training](#training)
  - [Export for FPGA](#export-for-fpga)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
  - [Feature Selection](#feature-selection)
  - [Label Mapping](#label-mapping)
- [Preprocessing](#preprocessing)
- [Artifacts Reference (for Hardware Team)](#artifacts-reference-for-hardware-team)
  - [model_spec.json](#model_specjson)
  - [model_weights.mem](#model_weightsmem)
  - [scaler.json](#scalerjson)
  - [har_model.keras](#har_modelkeras)
  - [weights_readable.txt](#weights_readabletxt)
  - [training_history.json](#training_historyjson)
- [Source Code Reference](#source-code-reference)
- [FPGA Implementation Notes](#fpga-implementation-notes)
- [Inference Pseudocode (for Hardware Translation)](#inference-pseudocode-for-hardware-translation)

---

## Results

| Metric              | Value   |
|---------------------|---------|
| Training Accuracy   | 95.3%   |
| Validation Accuracy | 95.7%   |
| Test Accuracy       | 92.7%   |
| Total Parameters    | 305     |
| Model Size          | 1.19 KB |
| Training Time (CPU) | ~10s    |

Per-class test accuracy:

| Class      | Accuracy | Correct / Total |
|------------|----------|-----------------|
| WALKING    | 100.0%   | 1387 / 1387     |
| SITTING    | 75.6%    | 371 / 491       |
| STANDING   | 87.0%    | 463 / 532       |
| LAYING     | 95.0%    | 510 / 537       |
| TRANSITION | N/A      | no test samples |

---

## Project Structure

```
har-fpga/
├── pyproject.toml                  # Project config, dependencies, entry points
├── README.md                       # This file
│
├── configs/
│   ├── features.json               # 19 selected feature names + indices
│   └── training.json               # Hyperparameters, label map, model config
│
├── src/har_fpga/                   # Source code (modular, translatable)
│   ├── __init__.py
│   ├── data.py                     # Download, extract, load UCI HAR, feature select, label merge
│   ├── preprocess.py               # Z-score normalization (fit / transform / save / load)
│   ├── model.py                    # Build Keras model, extract architecture spec
│   ├── train.py                    # Full training pipeline with CLI
│   ├── export.py                   # Export weights to .mem + JSON spec
│   └── infer.py                    # Run inference (single sample / batch / test eval)
│
├── artifacts/                      # Generated after training (SEND THIS TO HW TEAM)
│   ├── har_model.keras             # Saved Keras model (for reloading in Python)
│   ├── model_spec.json             # Layer-by-layer architecture spec (for FPGA)
│   ├── model_weights.mem           # All weights as IEEE-754 hex (for Verilog $readmemh)
│   ├── weights_readable.txt        # Human-readable weight dump (for inspection)
│   ├── scaler.json                 # Z-score normalization constants (mean + std per feature)
│   └── training_history.json       # Loss and accuracy per epoch
│
└── data/                           # Downloaded dataset (auto-cached, not committed)
    ├── UCI_HAR_Dataset.zip
    └── UCI HAR Dataset/            # Extracted dataset directory
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

This creates a `.venv`, resolves all dependencies (TensorFlow, NumPy, scikit-learn, requests),
and installs the project in editable mode. Takes about 30-60 seconds on first run.

---

## How to Run

### Training

```bash
# Default training (30 epochs, batch_size=64, lr=0.001)
uv run python -m har_fpga.train

# Custom hyperparameters
uv run python -m har_fpga.train --epochs 50 --batch-size 32 --lr 0.0005

# Force CPU (skip GPU detection)
uv run python -m har_fpga.train --no-gpu
```

On first run, the UCI HAR dataset (~61 MB) is automatically downloaded and cached in `data/`.
Subsequent runs reuse the cached data.

Training output:
- Prints model summary, epoch-by-epoch loss/accuracy, and final train/test accuracy.
- Saves all artifacts to `artifacts/`.

### Export for FPGA

```bash
# Export model_spec.json + model_weights.mem from the trained model
uv run python -m har_fpga.export

# Export from a specific model file
uv run python -m har_fpga.export --model path/to/model.keras
```

This reads the saved `.keras` model and produces:
- `model_spec.json` -- architecture for the hardware team
- `model_weights.mem` -- weights in IEEE-754 hex for `$readmemh`
- `weights_readable.txt` -- human-readable weight values

### Inference

```bash
# Predict a single sample (19 comma-separated raw feature values)
uv run python -m har_fpga.infer --sample "0.289,-0.020,-0.133,-0.995,-0.983,-0.914,-0.727,0.117,0.679,-0.998,-0.975,-0.960,-0.112,-0.065,0.029,-0.032,-0.563,-0.440,-0.202"

# Batch inference from a text file (one 19-value sample per line)
uv run python -m har_fpga.infer --file path/to/samples.txt

# Evaluate on the UCI HAR test split (prints overall + per-class accuracy)
uv run python -m har_fpga.infer --test
```

Single sample output example:
```
Predicted activity: LAYING (class 3)
Probabilities:
       WALKING: 0.0005
       SITTING: 0.0000
      STANDING: 0.0000
        LAYING: 0.9983  #######################################
    TRANSITION: 0.0012
```

---

## Model Architecture

A minimal 1D-CNN with **305 total parameters** (1.19 KB), designed for direct FPGA translation:

```
Input (19, 1)
  |
  v
Conv1D(filters=12, kernel_size=19, stride=1, padding='valid', activation='relu')
  |  -- output shape: (1, 12)
  |  -- kernel: [19 x 1 x 12] = 228 weights + 12 biases = 240 params
  v
Flatten
  |  -- output shape: (12,)
  v
Dense(units=5, activation='softmax')
  |  -- kernel: [12 x 5] = 60 weights + 5 biases = 65 params
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

No pooling layers, no dropout, no batch normalization -- intentionally kept minimal
so the hardware team can map it directly to RTL/HLS.

---

## Dataset

**UCI Human Activity Recognition Using Smartphones**

- Source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- 10,299 samples total (7,352 train + 2,947 test)
- Raw data: accelerometer + gyroscope from smartphone worn at the waist
- We use the **pre-extracted 561-feature vectors** provided by the dataset,
  then select 19 of them.

### Feature Selection

19 statistical features selected following the paper's criteria for minimal FPGA compute
with maximum discriminative power:

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

These cover translational dynamics (accelerometer), rotational dynamics (gyroscope),
jerk (time-derivative of acceleration), and vector magnitudes -- the most
discriminative signal components for activity recognition.

Full details in `configs/features.json`.

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
| Any other / transition | --        | TRANSITION  | 4         |

Full mapping in `configs/training.json`.

---

## Preprocessing

**Z-score normalization** is applied to each of the 19 features independently:

```
x_normalized = (x_raw - mean) / std
```

- `mean` and `std` are computed from the training set only.
- Saved to `artifacts/scaler.json` (19 mean values + 19 std values).
- **The FPGA must apply this same normalization before inference.**
- After normalization, the 19 values are reshaped from `(19,)` to `(19, 1)` to feed into Conv1D.

---

## Artifacts Reference (for Hardware Team)

After training, the `artifacts/` folder contains everything needed for FPGA implementation.
You can send this entire folder to the hardware team.

### model_spec.json

**What:** Complete layer-by-layer architecture specification.

**Contents:**
- `model_name`: `"har_1dcnn"`
- `total_params`: `305`
- `input_shape`: `[null, 19, 1]` (batch, features, channels)
- `output_shape`: `[null, 5]` (batch, classes)
- `layers`: Array of layer objects, each containing:
  - `name`, `type`, `output_shape`, `param_count`
  - `weight_shapes`: exact dimensions of each weight tensor
  - `config`: activation function, kernel size, padding, etc.

**Example layer entry (Conv1D):**
```json
{
  "name": "conv1d",
  "type": "Conv1D",
  "output_shape": [null, 1, 12],
  "param_count": 240,
  "weight_shapes": {
    "kernel": [19, 1, 12],
    "bias": [12]
  },
  "config": {
    "filters": 12,
    "kernel_size": [19],
    "strides": [1],
    "padding": "valid",
    "activation": "relu",
    "use_bias": true
  }
}
```

### model_weights.mem

**What:** All 305 model weights exported as IEEE-754 float32 hex values, one per line.

**Format:**
```
// kernel  shape=(19x1x12)  count=228
3dd05f49
3d77ddc6
be48b8e2
...
// bias  shape=(12)  count=12
3e838bac
...
```

- Each value is 8 hex characters = 32-bit IEEE-754 float.
- Lines starting with `//` are comments (section headers with tensor name and shape).
- Weight order is row-major (C order), matching `numpy.flatten()`.
- Compatible with Verilog `$readmemh`.

**Weight layout:**
1. `conv1d/kernel` -- shape `[19, 1, 12]` -- 228 values
2. `conv1d/bias` -- shape `[12]` -- 12 values
3. `dense_output/kernel` -- shape `[12, 5]` -- 60 values
4. `dense_output/bias` -- shape `[5]` -- 5 values

### scaler.json

**What:** Z-score normalization constants. **Must be applied to raw input before inference.**

**Contents:**
```json
{
  "mean": [0.2745, -0.0177, -0.1091, ...],   // 19 values
  "std":  [0.0703,  0.0408,  0.0566, ...]    // 19 values
}
```

**Usage:** For each feature `i`:
```
normalized[i] = (raw[i] - mean[i]) / std[i]
```

### har_model.keras

**What:** The full saved Keras model (weights + architecture + optimizer state).

**Usage:** Only needed if you want to reload the model in Python for further training,
fine-tuning, or running inference with `infer.py`. Not needed for FPGA implementation.

### weights_readable.txt

**What:** Human-readable version of all weights as decimal floats.

Small tensors (like biases) are printed value by value. Large tensors show
min/max/mean/std summary statistics.

### training_history.json

**What:** Per-epoch training metrics.

**Contents:** JSON object with keys `loss`, `accuracy`, `val_loss`, `val_accuracy`,
each mapping to an array of 30 float values (one per epoch).

Useful for plotting learning curves or verifying training convergence.

---

## Source Code Reference

Each module is self-contained and can be read/translated to C, Verilog, or HLS independently.

| File | Responsibility | Key Functions |
|------|---------------|---------------|
| `data.py` | Download UCI HAR, extract, load, select 19 features, remap labels | `load_har_data()`, `remap_labels()` |
| `preprocess.py` | Z-score normalization | `ZScoreScaler.fit()`, `.transform()`, `.save()`, `.load()` |
| `model.py` | Build Keras model, extract architecture spec | `build_model()`, `extract_model_spec()` |
| `train.py` | Full training pipeline with CLI | `train()`, `main()` |
| `export.py` | Export weights to .mem + JSON spec | `export_weights_mem()`, `export_spec_json()` |
| `infer.py` | Run inference (single/batch/test) | `predict_single()`, `predict_batch()`, `main()` |

---

## FPGA Implementation Notes

1. **Total compute per inference:**
   - Conv1D: 19 multiply-accumulate operations x 12 filters = 228 MACs + 12 bias adds + 12 ReLU
   - Dense: 12 x 5 = 60 MACs + 5 bias adds + softmax over 5 values
   - **Total: 288 MACs + 17 additions + 12 ReLU + 1 softmax(5)**

2. **Memory requirements:**
   - 305 weight parameters x 32 bits = 1,220 bytes (fits in BRAM easily)
   - 19 mean + 19 std values for normalization = 152 bytes
   - Total: ~1.4 KB

3. **Activations:**
   - **ReLU:** `y = max(0, x)` -- trivial in hardware (sign bit check)
   - **Softmax:** `y_i = exp(x_i) / sum(exp(x_j))` -- can use LUT-based exp approximation
     or skip softmax entirely and just take `argmax` if you only need the predicted class

4. **Fixed-point conversion:**
   - All weights are provided as float32 in the `.mem` file.
   - For fixed-point FPGA implementation, quantize weights to Q8.8, Q4.12, or similar
     format depending on your precision requirements.
   - Test quantized inference accuracy against the Python model before committing to a
     bit-width.

5. **Conv1D with kernel_size=19 on input length 19 with `valid` padding:**
   - This produces exactly **one** output position per filter.
   - Each filter computes: `sum(input[0:19] * kernel[0:19]) + bias`, then ReLU.
   - Effectively equivalent to a fully-connected layer from 19 inputs to 12 outputs.
   - This simplifies the hardware: no sliding window logic needed.

---

## Inference Pseudocode (for Hardware Translation)

This is the exact computation the FPGA must perform, step by step:

```
INPUT:  raw[19]          // 19 raw feature values from sensor preprocessing
OUTPUT: class_id         // integer 0-4

// --- Step 1: Z-score normalization ---
for i = 0 to 18:
    x[i] = (raw[i] - scaler_mean[i]) / scaler_std[i]

// --- Step 2: Conv1D (kernel=19, filters=12, valid, ReLU) ---
// Since kernel_size == input_length, there is exactly 1 output position.
// conv_kernel shape: [19][1][12], conv_bias shape: [12]
for f = 0 to 11:
    conv_out[f] = conv_bias[f]
    for i = 0 to 18:
        conv_out[f] += x[i] * conv_kernel[i][0][f]
    conv_out[f] = max(0, conv_out[f])    // ReLU

// --- Step 3: Flatten ---
// conv_out is already flat: shape (12,)

// --- Step 4: Dense output (12 -> 5, softmax) ---
// dense_kernel shape: [12][5], dense_bias shape: [5]
for c = 0 to 4:
    logit[c] = dense_bias[c]
    for f = 0 to 11:
        logit[c] += conv_out[f] * dense_kernel[f][c]

// --- Step 5: Argmax (skip softmax if you only need the class) ---
class_id = index of max(logit[0..4])

// --- Class mapping ---
// 0 = WALKING, 1 = SITTING, 2 = STANDING, 3 = LAYING, 4 = TRANSITION
```

---

## Training Configuration

All hyperparameters are in `configs/training.json`:

| Parameter        | Value                          |
|------------------|--------------------------------|
| Epochs           | 30                             |
| Batch Size       | 64                             |
| Learning Rate    | 0.001                          |
| Optimizer        | Adam                           |
| Loss Function    | Sparse Categorical Crossentropy|
| Validation Split | 20%                            |
| Random Seed      | 42                             |

---

## GPU Support

TensorFlow on native Windows dropped GPU support after v2.10. The installed `tensorflow==2.18`
runs on CPU only on Windows. Since the model has only 305 parameters, CPU training completes
in ~10 seconds -- GPU would provide no meaningful speedup for this model size.

If GPU training is needed for experimentation with larger models, use WSL2 with
`tensorflow[and-cuda]`.
