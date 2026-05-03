# 🍾 Bottle Detection Pipeline: Real-Time Damage Classification System

This project implements a **real-time, on-device bottle damage detection and classification system** using **YOLOv8** for bottle localization and a **ResNet50V2 TFLite model** for damage classification on the **Raspberry Pi 6**.

---

Download Dataset : https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/garvitsingh_iisc_ac_in/IgBZL7K_GBsFTKEEFdSSHR2iATd8kEWolyq1uzfXkRLRkwc?e=uzx9Id

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Repository Structure](#-repository-structure)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Hardware & Software Used](#-hardware--software-used)
- [Pipeline Architecture](#-pipeline-architecture)
  - [Custom Containment NMS](#custom-containment-nms-remove_overlapping_boxes)
- [Classifier: ResNet50V2 Training](#-classifier-resnet50v2-training-resnet50ipynb)
  - [Dataset & Split](#dataset--split)
  - [Augmentation Pipeline](#augmentation-pipeline)
  - [Figure 1: Augmented Training Samples](#figure-1-augmented-training-samples)
  - [Model Architecture](#model-architecture)
  - [Training: Two Phases](#training-two-phases)
  - [Figure 2: Training Curves (Phase 1 + Phase 2)](#figure-2-training-curves-phase-1--phase-2)
  - [Figure 3: Confusion Matrix](#figure-3-confusion-matrix)
  - [Quantization](#quantization)
  - [Figure 4: QAT Training Curves](#figure-4-qat-training-curves)
  - [Output Files](#output-files)
- [Performance Benchmarks](#-performance-benchmarks-raspberry-pi-6-arm64)
  - [Figure 5: Timing Breakdown Chart (best.pt baseline)](#figure-5-timing-breakdown-chart-bestpt-baseline)
  - [Figure 6: QAT Model Results — 5× YOLO Speedup](#figure-6-qat-model-results--5-yolo-speedup)
  - [Figure 7: INT8 Model Results](#figure-7-int8-model-results)
  - [Memory Usage](#memory-usage-rpi6-8-gb-lpddr5)
  - [Figure 8: Memory Usage Profile](#figure-8-memory-usage-profile)
- [Usage Modes](#-usage-modes)
  - [Image Mode](#image-mode)
  - [Folder Mode](#folder-mode)
  - [Camera Mode — Photo Booth Workflow](#camera-mode--photo-booth-workflow)
  - [Figure 9: Live Detection Output](#figure-9-live-detection-output)
- [Data Structures](#-data-structures)
- [Raspberry Pi Setup](#-raspberry-pi-setup)
- [Troubleshooting](#-troubleshooting)
- [V2 Roadmap](#-v2-roadmap-scaling-to-real-time-conveyor-speeds)
- [Team](#-team)

---

## ✨ Highlights

* **Fully offline Edge AI pipeline** — All detection and classification runs locally on the Raspberry Pi 6 with **no cloud or internet dependency** for core functionality.
* **Two-stage architecture** — YOLOv8 handles fast spatial bottle localization across the full frame; ResNet50V2 provides high-fidelity per-crop defect analysis with richer texture features than a single-model head alone.
* **Tiny but capable classifier** — ResNet50V2 INT8 quantized TFLite model (~25 MB), running at **~277 ms per bottle** on RPi6 ARM64 with the XNNPACK delegate.
* **5× YOLO speedup via model switch** — Switching from `best.pt` (custom fine-tuned) to `yolov8n.pt` reduced YOLO inference from ~1550 ms to **~308 ms** with no accuracy configuration change. Note: the CLI defaults to `yolov8s.pt`; pass `--yolo-model yolov8n.pt` explicitly for maximum speed on RPi.
* **Democratises QC for FMCG** — Traditional industrial vision systems cost ₹50,000+. This pipeline runs on a **₹6,000 Raspberry Pi**, making automated quality control accessible to small and mid-scale manufacturers.

---

## 📁 Repository Structure

```text
bottle_pipeline/
├── bottle_pipeline.py        # Main entry point — 3 operating modes
├── resnet50.ipynb            # Classifier training notebook (Colab/Kaggle)
├── setup_rpi5.sh             # 7-step installation script for Raspberry Pi
├── requirements.txt          # Python dependencies
└── models/
    ├── yolov8s.pt            # YOLOv8s detection weights (default)
    ├── yolov8n.pt            # YOLOv8n detection weights (recommended for RPi speed)
    ├── *.tflite              # ResNet50V2 damage classifier (INT8 / QAT / FP32)
    └── labels.txt            # Class labels: damaged, non_damaged
```

---

## 🚥 Problem Statement

Packaging quality control is a critical step in FMCG manufacturing. Damaged bottles reaching consumers lead to product spoilage, customer complaints, and brand damage. Current automated inspection solutions require:

* Expensive industrial camera rigs
* Cloud API dependencies and proprietary software licences
* Vendor lock-in with per-unit pricing

This project addresses that gap with a **fully open-source, offline Edge AI pipeline** that can be deployed on low-cost hardware — enabling quality control for small and mid-scale producers who previously had no viable option.

---

## 🎯 Project Objectives

The main objective is to develop a production-grade bottle inspection module that:

* Detects all bottles in a frame using a real-time object detector
* Classifies each detected bottle as `damaged` or `non_damaged` using a fine-tuned CNN
* Provides immediate feedback via annotated frames (green = ok, red = damaged) and a JSON result log
* Runs **entirely on-device** on a Raspberry Pi, with **no cloud dependency**

The prototype demonstrates an **end-to-end Edge pipeline**:

> Dataset preparation → Transfer learning (ResNet50V2) → QAT quantization → TFLite export → On-device deployment on RPi6

---

## 🔧 Hardware & Software Used

### Hardware Required

* 🧠 **Raspberry Pi 6** (ARM64, 8 GB LPDDR5)
* 📷 **Pi Camera Module** (or USB webcam for camera mode)
* 🔌 **USB power supply / battery bank** for portable deployment

### Software & Tools Used

* 🐍 **Python 3** with `ultralytics`, `tflite-runtime`, `opencv-python-headless`, `numpy`, `pillow`
* 🔥 **PyTorch (ARM CPU build)** — for running YOLOv8 inference via Ultralytics
* 📦 **TensorFlow Lite** — lightweight inference engine for the ResNet50V2 classifier
* 🧪 **TensorFlow / Keras** — model training, evaluation, and TFLite export (run on Colab/Kaggle)
* 🔬 **TensorFlow Model Optimization Toolkit** — for Quantization Aware Training (QAT)
* 📊 **Matplotlib / Seaborn** — training curve and confusion matrix plots
* 🔢 **scikit-learn** — `classification_report`, `confusion_matrix`, `compute_class_weight`
* 🖥️ **Google Colab / Kaggle** — cloud GPU environment for training

---

## 🏗️ Pipeline Architecture

The pipeline uses a **two-stage architecture** (see PPT Slide 2 — Pipeline Overview):

| Stage | Tool | Role |
|---|---|---|
| 1 — Detection | YOLOv8 | Localize all bottles in the frame with configurable confidence threshold (default 0.10) |
| 1 — NMS Filter | Custom containment NMS | Remove large wrapper boxes that contain smaller individual bottle boxes |
| Bridge | OpenCV | Crop each bottle with 20 px padding, preserving context for the classifier |
| 2 — Classify | ResNet50V2 TFLite | Predict `damaged` / `non_damaged` per crop; softmax probabilities logged |
| 2 — Log Output | JSON | Save annotated image (green = ok, red = damaged), per-bottle crop files, structured JSON |

**Why two stages?** YOLO excels at robust, fast spatial detection across variable bottle counts and positions. ResNet provides richer texture-level feature extraction per crop — better suited to detecting subtle surface damage than using the YOLO classification head alone.

### Custom Containment NMS (`remove_overlapping_boxes`)

Standard IoU-based NMS fails in two real scenarios this pipeline encounters. The custom NMS is **containment-based** and handles both (see PPT Slide 3 for the logic diagram):

1. **Duplicate box around a single bottle** — YOLO draws a tight box and a slightly larger wrapper box around the same bottle
2. **Large wrapper box spanning multiple bottles** — YOLO draws one big box around a group of 2+ bottles alongside correct individual boxes

**Logic:** Boxes are sorted by area (largest first). A large box is removed if it contains **1 or more** smaller boxes with ≥ 80% overlap (`containment_threshold=0.80`, `iou_threshold=0.30`). Only the tightest individual-bottle boxes are forwarded to the classifier.

---

## 🧠 Classifier: ResNet50V2 Training (`resnet50.ipynb`)

The damage classifier is trained in a Jupyter notebook on Colab/Kaggle using a two-phase transfer learning approach. The notebook follows a 14-step pipeline from data loading to TFLite export.

### Dataset & Split

* **Classes:** `damaged` | `non_damaged`
* **Split:** 70% train / 15% validation / 15% test
* Splits are **deterministic** via `SEED=42` — files sorted before shuffle to guarantee reproducibility across runs
* A **data leakage check** is performed post-split: `train_files.intersection(test_files)` must return zero overlap
* Class counts verified at startup: any missing class folder prints an error before training begins

### Augmentation Pipeline

Training images are augmented on-the-fly using a `tf.keras.Sequential` augmentation block applied only during training (`training=True`):

| Transform | Parameter | Purpose |
|---|---|---|
| `RandomFlip` | horizontal | Mirror-invariance |
| `RandomRotation` | ±10° (0.10) | Slight tilt tolerance |
| `RandomZoom` | 10% | Scale variation |
| `RandomBrightness` | 20% | Lighting variation |
| `RandomContrast` | 15% | Exposure variation |
| `GaussianNoise` | σ = 0.05 | Simulate sensor noise |

ResNet50V2 preprocessing (`preprocess_input`) is applied **after** augmentation, scaling pixel values from `[0, 255]` to `[-1, 1]` as required by the backbone. All three splits use `AUTOTUNE` prefetching for pipeline efficiency.

### Figure 1: Augmented Training Samples

> **`augmented_samples.png`** — 3×3 grid of augmented training images, saved during Step 5 of `resnet50.ipynb`. Shows the combined visual effect of RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, and GaussianNoise on real bottle crops.

```
OUTPUT_DIR/augmented_samples.png
```

### Model Architecture

* **Backbone:** ResNet50V2 pretrained on ImageNet (`include_top=False`, `weights='imagenet'`)
* **Input:** 224 × 224 × 3 RGB

**Classification head:**

```
GlobalAveragePooling2D
BatchNormalization
Dense(256, activation='relu', kernel_regularizer=L2(1e-4))
Dropout(0.57)
Dense(128, activation='relu', kernel_regularizer=L2(1e-4))
Dropout(0.30)
Dense(2, activation='softmax')   ← damaged | non_damaged
```

Total trainable parameters: classification head only in Phase 1; head + top 30 ResNet layers in Phase 2. See PPT Slide 4 for the inference flow diagram.

### Training: Two Phases

**Phase 1 — Frozen base (up to 15 epochs)**

The ResNet50V2 backbone is fully frozen (`base_model.trainable = False`). Only the classification head trains.

* Optimizer: `Adam(lr=1e-3)`
* Loss: `CategoricalCrossentropy(label_smoothing=0.1)`
* Class weights via `compute_class_weight('balanced')` — handles imbalanced damaged/non-damaged counts
* Callbacks: `EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`, `ModelCheckpoint(save_best_only=True)`

**Phase 2 — Fine-tuning top 30 layers (up to 20 epochs)**

The top 30 layers of the backbone are unfrozen (`layer.trainable = True` for `base_model.layers[-30:]`).

* Optimizer: `Adam(lr=5e-6)` — deliberately low to avoid destroying pretrained weights
* Same loss and callbacks; `EarlyStopping(patience=7)` gives more patience as improvement is slower
* Best weights restored automatically at end of training

### Figure 2: Training Curves (Phase 1 + Phase 2)

> **`training_curves.png`** — 3-panel figure saved during Step 9 of `resnet50.ipynb` at 150 DPI. **Panel 1:** Train vs. Val accuracy across all epochs. **Panel 2:** Train vs. Val loss across all epochs. **Panel 3:** Train Precision and Recall across all epochs. A vertical dashed line marks the Phase 1 → Phase 2 boundary (fine-tune start). Title: *"ResNet50V2 — Training Curves"*.

```
OUTPUT_DIR/training_curves.png
```

### Figure 3: Confusion Matrix

> **`confusion_matrix.png`** — Seaborn heatmap saved during Step 10 of `resnet50.ipynb`. Test-set confusion matrix with raw counts, axes: true label (y) vs. predicted label (x). Classes: `damaged`, `non_damaged`. Title: *"Confusion Matrix — Test Set (ResNet50V2)"*.

```
OUTPUT_DIR/confusion_matrix.png
```

### Quantization

Two quantized variants are exported for edge deployment:

**Post-Training INT8 Quantization (Step 11)**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen   # 300 calibration batches
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32   # float I/O avoids mismatch on RPi
converter.inference_output_type = tf.float32
```

**Quantization Aware Training — QAT (Step 12)**

The classification head is wrapped with `tfmot.quantization.keras.quantize_model` and the full model re-trained for up to 30 epochs at `lr=1e-5` with `EarlyStopping(patience=5)`. QAT simulates quantization noise during training, yielding **better accuracy at INT8 precision** than post-training quantization. On RPi6, QAT also runs faster (~252 ms) than the post-training INT8 model (~277 ms) because XNNPACK's FP32 SIMD kernels outperform its INT8 path on the ARM Cortex-A76 — **QAT is the recommended deployment format.**

A FP32 TFLite model is also exported (no optimizations) for debugging and accuracy comparison.

### Figure 4: QAT Training Curves

> **`qat_training_curves.png`** — 2-panel figure saved during the QAT training cell of `resnet50.ipynb`. **Panel 1:** QAT training vs. validation accuracy across all QAT epochs. **Panel 2:** QAT training vs. validation loss across all QAT epochs.

```
OUTPUT_DIR/qat_training_curves.png
```

### Output Files

| File | Step | Description |
|---|---|---|
| `bottle_classifier_resnet50v2_fp32.keras` | 8 | Full FP32 Keras model, best Phase 2 weights |
| `best_phase1.keras` | 7 | Best checkpoint from Phase 1 (frozen base) |
| `best_phase2.keras` | 8 | Best checkpoint from Phase 2 (fine-tuned) |
| `bottle_classifier_resnet50v2_int8.tflite` | 11 | INT8 post-training quantized TFLite (~22 MB) |
| `bottle_classifier_resnet50v2_fp32.tflite` | 11 | FP32 TFLite (no quantization, for debug/comparison) |
| `bottle_classifier_resnet50v2_qat.tflite` | 12 | QAT TFLite — **recommended for deployment** |
| `labels.txt` | 13 | Class label list (`damaged`, `non_damaged`) |
| `training_curves.png` | 9 | Figure 2: Accuracy / Loss / Precision+Recall (all epochs) |
| `confusion_matrix.png` | 10 | Figure 3: Test set confusion matrix |
| `augmented_samples.png` | 5 | Figure 1: 3×3 grid of augmented training samples |
| `qat_training_curves.png` | 12 | Figure 4: QAT accuracy and loss curves |

---

## 📊 Performance Benchmarks (Raspberry Pi 6, ARM64)

All benchmarks run on RPi6 ARM64 with XNNPACK delegate. Confidence threshold: 0.10.

### Figure 5: Timing Breakdown Chart (best.pt baseline)

> **PPT Slide 7** — Bar chart benchmarked on RPi6 using the original `best.pt` FP32 YOLO model. Six bars: YOLO 1 bottle (1550 ms), YOLO 2 bottles (885 ms), ResNet ×1 (280 ms), ResNet ×2 (505 ms), Total 1 bottle (2016 ms), Total 2 bottles (1559 ms). Note: *"Includes model load on first run. Subsequent calls are faster."*

### Inference Timing

| Configuration | YOLO | ResNet/bottle | Total (1 bottle) | Total (2 bottles) |
|---|---|---|---|---|
| FP32 (`best.pt`) | ~1550 ms | ~252 ms | **~2016 ms** | **~1559 ms** |
| QAT (`yolov8n.pt`) | ~308 ms | ~252 ms | **~2076 ms** | **~1351 ms** |
| INT8 (`yolov8n.pt`) | ~308 ms | ~277 ms | **~2245 ms** | **~1424 ms** |

### Figure 6: QAT Model Results — 5× YOLO Speedup

> **PPT Slide 12** — Shows the before/after YOLO banner: `best.pt` ~1550 ms → `yolov8n.pt` ~308 ms (**5× FASTER**). Two benchmark runs: Run 01 (7 bottles, YOLO 307.6 ms, ResNet QAT ~252 ms, total 2076 ms) and Run 02 (4 bottles, YOLO 310.9 ms, ResNet QAT ~260 ms, total 1351 ms). Key win note: switching the detection model from `best.pt` to `yolov8n.pt` achieved the speedup with no accuracy configuration change.

### Figure 7: INT8 Model Results

> **PPT Slide 13** — Three-column comparison header: FP32 best.pt (YOLO ~1550 ms, total 2016 ms), QAT yolov8n (YOLO ~308 ms, total 2076 ms), INT8 yolov8n (YOLO ~308 ms, total **2245 ms** — slower than QAT). Same two benchmark runs below. Observation panel explains the counterintuitive result: XNNPACK on RPi6's ARM Cortex-A76 does not accelerate INT8 convolutions as efficiently as FP32. QAT keeps weights in float32 at runtime — XNNPACK's highly-tuned FP32 SIMD kernels outperform its INT8 path on this SoC. INT8 wins on memory bandwidth, not compute throughput.

### Memory Usage (RPi6, 8 GB LPDDR5)

| Component | RAM |
|---|---|
| YOLOv8n model weights | ~12 MB |
| ResNet50V2 TFLite INT8 | ~25 MB |
| OpenCV frame buffer | ~50 MB |
| Python runtime + libs | ~200 MB |
| XNNPACK delegate cache | ~30 MB |
| OS + idle processes | ~348 MB |
| **Pipeline peak RSS** | **~665 MB (8.1% of 8 GB)** |

### Figure 8: Memory Usage Profile

> **PPT Slide 15** — Horizontal utilization bar showing pipeline RSS (~665 MB, 8.1%) vs. total 8 GB with ~7.4 GB free. Right panel breaks down RAM by component. Headroom section projects V2 expansion: multi-camera ×4 feeds (~2.2 GB total), 640×480 resolution input (+80 MB), Edge TPU runtime overhead (+50 MB), remaining free after V2 (>5 GB). Note: *"Only 8.1% of available RAM consumed. The pipeline can scale to 10+ simultaneous models, higher-resolution inputs, or multi-camera feeds without requiring a hardware upgrade."*

---

## 🚀 Usage Modes

The pipeline supports three operating modes via the `--mode` flag. Entry point: `bottle_pipeline.py`. See PPT Slide 6.

### Image Mode

Single-shot inspection. Best for validation and debugging.

```bash
python bottle_pipeline.py \
  --mode image \
  --input photo.jpg \
  --confidence 0.10 \
  --output results/
```

**Outputs:** Annotated image with green (ok) / red (damaged) bounding boxes and `{class}: {confidence}` labels, per-bottle crop files, JSON result log.

### Folder Mode

Bulk processing. Logs a summary across all images at the end.

```bash
python bottle_pipeline.py \
  --mode folder \
  --input images/ \
  --output batch_results/
```

**Outputs:** Batch JSON summary, damage rate across all images, avg processing time per image. Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

### Camera Mode — Photo Booth Workflow

Designed for offline QC inspection stations. Runs a repeating 3-phase loop — avoids the RPi CPU being continuously maxed between captures.

```bash
python bottle_pipeline.py \
  --mode camera \
  --camera 0 \
  --no-display        # for headless RPi deployments
```

**Loop phases (repeating):**
1. **Live preview + 5s countdown** — shows a live feed with *"Capturing in: N"* overlay so the operator can position bottles
2. **Capture & process** — freezes the last frame, runs YOLO + ResNet, draws annotated results. Camera buffer set to 1 frame to minimise stale-frame lag.
3. **Display results for 10s** — shows annotated frame with per-bottle labels; operator reads result before next cycle begins

**Outputs:** Annotated result per capture, running session totals (total bottles / total damaged / damage rate). Keys: `q` = quit, `s` = save current annotated frame to disk. Session summary printed to terminal and saved as JSON on exit.

### Figure 9: Live Detection Output

> **PPT Slide 11** — Real inference output on 4 bottles (Bisleri and similar PET bottles). Pipeline correctly identifies 2 as `Not Damaged` (61%, 68%) and 2 as `Damaged` (68%, 72%). All 4 have individual bounding boxes. Summary overlay top-left: *"Total: 4 | Damaged: 2 | Non Damaged: 2"*. Boxes appear green for both classes in this screenshot — in the actual runtime, damaged bottles get red boxes.

---

## 🧱 Data Structures

Results are returned as typed Python `@dataclass` objects for type safety and easy JSON serialization. See PPT Slide 5.

**`BottleDetection`** — one detection + classification result per bottle:

```python
@dataclass
class BottleDetection:
    bottle_id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    detection_confidence: float
    damage_class: str                  # 'damaged' | 'non_damaged'
    damage_confidence: float
    crop_path: Optional[str]           # path to saved crop file
```

**`PipelineResult`** — full image result, serializable via `json.dump(asdict(result))`:

```python
@dataclass
class PipelineResult:
    image_path: str
    timestamp: str
    total_bottles: int
    damaged_count: int
    not_damaged_count: int
    detection_time_ms: float
    classification_time_ms: float
    total_time_ms: float
    detections: List[BottleDetection]
```

**JSON Output Sample (folder mode, full summary):**
```json
{
  "summary": {
    "total_images": 12,
    "total_bottles": 48,
    "total_damaged": 11,
    "total_not_damaged": 37,
    "damage_rate": "22.9%"
  },
  "results": [
    {
      "total_bottles": 3,
      "damaged_count": 1,
      "total_time_ms": 245.3,
      "detections": [...]
    }
  ]
}
```

Crop files are saved as `{image_name}_bottle_{id}_{class}.jpg`.

---

## ⚙️ Raspberry Pi Setup

Run `setup_rpi5.sh` for automated 7-step installation. See PPT Slide 8.

```bash
# 1. System update
sudo apt update && sudo apt upgrade -y

# 2. System dependencies
apt install python3-pip python3-venv libopencv-dev libatlas-base-dev cmake git

# 3. Virtual environment
python3 -m venv ~/bottle_pipeline_venv

# 4. TFLite runtime
pip install tflite-runtime

# 5. PyTorch ARM CPU build
pip install torch torchvision --index-url .../whl/cpu

# 6. Ultralytics YOLO
pip install ultralytics

# 7. Other dependencies
pip install numpy opencv-python-headless pillow
```

**Required file layout:**
```
bottle_pipeline/
├── bottle_pipeline.py
├── setup_rpi5.sh
├── requirements.txt
└── models/
    ├── yolov8s.pt          ← CLI default; swap for yolov8n.pt for speed
    ├── *.tflite
    └── labels.txt
```

**Transfer models from PC:**
```bash
scp yolov8n.pt pi@raspberrypi:~/bottle_pipeline/models/
scp *.tflite pi@raspberrypi:~/bottle_pipeline/models/
```

**Activate and run:**
```bash
source ~/bottle_pipeline_venv/bin/activate
cd ~/bottle_pipeline
# CLI default is yolov8s.pt — pass yolov8n.pt explicitly (all benchmarks use yolov8n)
python bottle_pipeline.py --mode camera --yolo-model models/yolov8n.pt
```

---

## 🔍 Troubleshooting

See PPT Slide 9 for the full fault tree.

| Error | Fix |
|---|---|
| `No module named tflite_runtime` | `pip install tflite-runtime`. Use `tflite-runtime` on RPi, not full TensorFlow. Fallback: `import tensorflow as tf` |
| `Could not open camera` | `ls /dev/video* && raspi-config` — verify `/dev/video*` exists and enable camera interface for Pi Camera Module |
| `YOLO model loading error` | `pip install torch torchvision --index-url .../cpu` — ensure PyTorch ARM CPU build; download index must specify `/whl/cpu` |
| Out of memory (OOM) | Use `--yolo-model yolov8n.pt` — YOLOv8n uses far less RAM than YOLOv8s; process images one-at-a-time in folder mode |
| Camera feed lags / stale frames | Camera mode sets `CAP_PROP_BUFFERSIZE=1` — if lag persists, add a `cap.read()` discard call before the capture step |

### Speed Tuning

| Knob | Recommendation |
|---|---|
| YOLO model size | `yolov8n.pt` > `yolov8s.pt` — 5× faster on RPi6 |
| Resolution | 320×240 vs 640×480 → ~2× faster; camera mode defaults to 640×480 |
| Quantization | QAT TFLite is the fastest and most accurate option on RPi6 |
| Confidence threshold | Higher threshold → fewer crops forwarded → faster overall |
| Batch size | Process 1 image at a time on RPi (pipeline already does this) |

---

## 🗺️ V2 Roadmap: Scaling to Real-Time Conveyor Speeds

See PPT Slide 14.

| Phase | Status | Achievements / Plan |
|---|---|---|
| Phase 1 — INT8 Quantization | ✅ Done | ResNet50V2 INT8v3 TFLite deployed; QAT validated at ~252 ms/bottle (QAT), ~277 ms/bottle (INT8); XNNPACK delegate enabled |
| Phase 2 — YOLOv8n Nano Switch | ✅ Done | Switched `best.pt` → `yolov8n.pt`; YOLO 1550 ms → 308 ms (5× speedup measured on RPi6); conf=0.10 accuracy within acceptable range |
| Phase 3 — Edge TPU Integration | 🔵 Planned | Coral USB Accelerator or Dev Board; INT8 model compiled for Edge TPU; target <50 ms/bottle end-to-end; eliminates CPU bottleneck entirely |
| Phase 4 — Conveyor Integration | 🔵 Planned | GPIO-triggered capture via sensor; Pi Camera 3 (rolling shutter fix); continuous stream zero-buffer mode; alert output → PLC / reject actuator |

**Current V1:** ~2–3.5 s/tray (batch processing, best.pt FP32, RPi CPU only — suitable for offline QC inspection workflow).
**Target V2:** ~50 ms/bottle on continuous conveyor stream with industrial-grade zero missed-defect guarantee.

---

## 👥 Team

* **Bottle Pipeline Team** — 2026

For questions, feedback, or collaboration, please open an issue in this repository.
