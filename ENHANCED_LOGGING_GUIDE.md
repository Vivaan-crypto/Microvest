# Enhanced Lightning Logs - Quick Reference

## What You're Now Logging 📊

Your improved training pipeline now logs **11 metrics** (up from 2) plus **3 different visualizations**.

---

## Scalar Metrics (Numbers)

### Training Step Metrics (Logged Every 10 Steps)
```
train/loss             Average cross-entropy loss on batch
train/accuracy         % of predictions that are correct
train/f1               F1 score (balance of precision/recall)
train/precision        How often model is right when it predicts
train/recall           What % of actual positives it catches
```

**Why these separately?**
- `loss` tells you if learning is happening
- `accuracy` is intuitive (80% accuracy = good?)
- `f1/precision/recall` separate different failure modes
  - High precision, low recall: too conservative
  - Low precision, high recall: too aggressive

### Validation Epoch Metrics (Logged Every Epoch)
```
val/loss               Same metrics, but on unseen validation data
val/accuracy
val/f1                 
val/precision
val/recall             ← Most important for detecting overfitting
```

### Optimizer Metric
```
learning_rate          Adam's learning rate per epoch
                       Should stay constant unless you modify it
```

---

## Visual Metrics (Plots)

### 1️⃣ Confusion Matrix
**Logged:** Every 100 training steps + every validation epoch

**What it shows:**
```
        Predicted
        Down Flat Up
Actual
Down    [80  10  5]   ← Good: most Down predictions on Down actual
Flat    [15  50  10]  ← Good: most Flat predictions on Flat actual
Up      [5   10  85]  ← Good: most Up predictions on Up actual
```

**What to look for:**
- ✅ High diagonal values = good predictions
- ⚠️ Off-diagonal clumps = systematic errors
- Example: Down often predicted as Flat = model confused

### 2️⃣ Confidence Distribution
**Logged:** Every 100 training steps + every validation epoch

**What it shows:**
```
Histogram of model's confidence (highest softmax probability)

Example: 
- All confidences near 1.0 → Model is very confident
- Spread across 0-1 → Model is uncertain
- Mean shown as red line for reference
```

**What to look for:**
- ✅ Distribution shifts right over time (more confident)
- ✅ Validation more uncertain than training (normal)
- ⚠️ Val stays low while train goes to 1.0 = overfitting
- ⚠️ Never improves = model not learning

### 3️⃣ Per-Class Metrics (Validation Only)
**Logged:** Every validation epoch

**3 subplots side-by-side:**
```
Precision by Class    Recall by Class    F1 by Class
(Down/Flat/Up)        (Down/Flat/Up)     (Down/Flat/Up)

Shows if model:
- Struggles with one class (low bar for that class)
- Has imbalanced performance (some 0.9, others 0.5)
```

**Example interpretation:**
```
Down:  Precision 0.85, Recall 0.80, F1 0.82
Flat:  Precision 0.70, Recall 0.65, F1 0.67  ← Struggles here
Up:    Precision 0.88, Recall 0.92, F1 0.90
```
→ Model needs help with Flat predictions

---

## How Metrics Relate to Directionality

Your task: Predict if stock goes **Down (-1) / Flat (0) / Up (1)**

### What Each Metric Tells You

| Metric | Interpretation for Direction Prediction |
|--------|------------------------------------------|
| **Accuracy** | % of time you correctly predict the direction |
| **F1 Score** | Balanced score (best single metric to watch) |
| **Precision** | When I say "Up", am I usually right? |
| **Recall** | Do I catch most of the actual "Up" days? |
| **Loss** | Overall confidence in the decisions |

### Example Scenario
```
Epoch 50:
  val/accuracy: 0.65 (65% correct)
  val/f1: 0.62
  val/confusion_matrix shows:
    - Down: 70% correct
    - Flat: 50% correct (problem!)
    - Up: 75% correct
    
Interpretation: Model is struggling to identify "Flat" markets
Action: May need more training data or better features for sideways movement
```

---

## Reading TensorBoard

### Navigate to your logs:
```bash
tensorboard --logdir lightning_logs/
# Open http://localhost:6006
```

### What you'll see in TensorBoard UI:

**SCALARS tab:**
- Multiple lines (one per metric)
- X-axis: training steps/epochs
- Y-axis: metric value
- Hover for exact values

**IMAGES tab:**
- Confusion matrices from different training stages
- Confidence histograms over time
- Per-class metrics bar charts

### Healthy training pattern:

```
80 ├─────╱─────────  ← val/accuracy improving
   │    ╱
60 ├──╱──────────────
   │╱
40 └─────────────────
    0      50      100 epochs
    
✓ Curves improving over time = learning
✓ Train > Val = expected
✓ Both plateauing = convergence or early stopping
```

### Red flags:

```
100├──────────────────  ← train/accuracy stuck at ceiling
80 ├─────────┐
   │         └────────  ← val/accuracy decreasing
60 ├──────────────────
   │
✗ Train far above val = overfitting
✗ Val decreasing = early stopping triggered or model failing
```

---

## Custom Thresholds to Monitor

### For Stock Directionality:
```
✅ Good:      val/f1 > 0.55 (better than random 0.33)
✅ Very Good: val/f1 > 0.65 (65% win rate in class accuracy)
✅ Excellent: val/f1 > 0.75 (very reliable predictions)

⚠️  Concerning:  val/f1 < 0.40 (doesn't beat chance baseline)
⚠️  Bad:         val/f1 plateaus for 10+ epochs
⚠️  Failing:     val/accuracy < 0.50 (worse than two-class random)
```

### Early Stopping Trigger:
```
Stops training if validation F1 doesn't improve by 0.001
for 20 consecutive epochs

This prevents:
✓ Wasting compute time
✓ Training past the best model
✓ Severe overfitting
```

---

## Example: Interpreting Your First Run

Day 1, Epoch 1:
```
train/loss: 1.098
train/f1: 0.33           ← All predictions random class
train/accuracy: 0.33     ← Chance level (3 classes)
val/f1: 0.29
val/accuracy: 0.32       ← Expected: model hasn't learned yet
```

Day 1, Epoch 50:
```
train/loss: 0.654
train/f1: 0.72           ← Getting good on training
train/accuracy: 0.75
---
val/f1: 0.58             ← Good: generalizing
val/accuracy: 0.65       ← Model learning patterns
```

Day 1, Epoch 100:
```
train/loss: 0.312
train/f1: 0.89           ← Excellent on training
train/accuracy: 0.92     ← Very confident training
---
val/f1: 0.61             
val/accuracy: 0.68       
(unchanged from epoch 50)  ← Plateau → early stopping triggers
```

**Conclusion:** Best model was at epoch 50 (automatically saved in checkpoints/)

---

## TensorBoard Commands

```bash
# Start TensorBoard (run in terminal)
cd C:\GitHub\Microvest\src\PredictionApp
tensorboard --logdir lightning_logs/ --port 6006

# Open in browser: http://localhost:6006

# To see multiple runs (compare different training sessions):
tensorboard --logdir lightning_logs/

# Each run appears as separate row in scalars tab
```

---

## Column Reference: What to Export

If you want to export metrics for reporting:

### Export from TensorBoard:
- Right-click any plot → "Download as CSV"
- Gives you step-by-step values

### Or from checkpoint:
```python
import torch
import json

# Load hparams
hparams = json.load(open("lightning_logs/stock_prediction_model/version_0/hparams.yaml"))

# Get best F1 from model filename
# "best-model-epoch_02-val_f1_0.658"
```

---

## Summary: What Changed

| Aspect | Before | After | Why |
|--------|--------|-------|-----|
| Metrics | 2 (loss, f1) | 11 scalar + 3 plots | Better diagnosis |
| Training logs | Every step | Aggregated per epoch | Cleaner dashboard |
| Checkpointing | None | Best + Latest saved | Can revert if needed |
| Early stopping | No | Yes (patience=20) | Save compute time |
| Class insight | Overall F1 only | Per-class breakdown | Identify weak classes |
| Confidence track | Not tracked | Logged + visualized | Detect uncertainty issues |

---

Generated: June 10, 2026  
All metrics available in TensorBoard at: `lightning_logs/stock_prediction_model/version_X/`

