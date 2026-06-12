# Gradient Logging - Quick Guide

## What's Now Being Logged

### Overall Gradient Health
```
train/gradient_norm
  └─ L2 norm of all gradients combined
  └─ Tells you: How "steep" the training landscape is
```

### Per-Layer Gradient Norms
```
gradients/input_linear
  └─ Input embedding layer gradients

gradients/transformer_layer0
  └─ First transformer encoder layer
  └─ Shows early feature learning

gradients/transformer_layer5
  └─ Last transformer encoder layer
  └─ Shows deep feature refinement

gradients/output_head
  └─ Classification head gradients
  └─ Shows final output layer training
```

**Total: 5 gradient metrics logged every training step**

---

## What to Look For in TensorBoard

### Healthy Gradient Pattern
```
gradient_norm over time:

1.0 ├─────╱────────────
    │    ╱
0.5 ├──╱──────────────
    │╱
0.0 └─────────────────
    0      50      100 steps

✓ Stable around 0.1-1.0
✓ Decreases gradually as loss minimizes
✓ No sudden spikes or crashes
```

### Red Flags

#### 1. Vanishing Gradients
```
gradient_norm:

0.1 ├──────────────────
    │
0.01├──────────────────
    │
10⁻⁵└──────────────────  ← Approaching zero = dead network
    0      50      100
    
⚠️ Gradients → 0 = parameters stop updating
⚠️ Common in: Deep networks, poor initialization
✓ Fix: Reduce learning rate or add skip connections
```

#### 2. Exploding Gradients
```
gradient_norm:

1000├────────────────
    │
 100├────────────────
    │
  10├────────────────
    │  (gradient clipping kicks in)
    0      50      100
    
⚠️ Huge spikes = model becoming unstable
✓ Already handled: gradient_clip_val=1.0 in trainer
```

#### 3. Layer-Specific Issues
```
If gradients/input_linear ≈ 0 but gradients/output_head is normal:
  → Information bottleneck in input layer
  → Consider: Larger embedding dimension or pre-training

If gradients/transformer_layer0 >> gradients/transformer_layer5:
  → Early layers learning much faster than deep layers
  → Consider: Layer normalization or learning rate scheduling
```

---

## How Gradients Relate to Training

```
Loss decreases
    ↓
Gradients computed (backward pass)
    ↓
●── Large gradients = steep descent (fast learning)
●── Small gradients = flat region (slow learning)
●── Zero gradients   = stuck (not learning)
    ↓
Parameters updated by: new_param = param - lr × gradient
    ↓
Next epoch starts
```

### Example Interpretation

```
Epoch 10:
  train/loss: 1.50 → train/gradient_norm: 0.85
  → Loss is high, gradients are substantial → should improve quickly

Epoch 50:
  train/loss: 0.45 → train/gradient_norm: 0.15
  → Loss lower, gradients smaller → approaching convergence

Epoch 100:
  train/loss: 0.35 → train/gradient_norm: 0.02
  → Loss plateauing, tiny gradients → learning has mostly saturated
```

---

## Compare With Loss For Debugging

### Scenario 1: Good Training
```
Loss decreasing    ✓
Gradients stable   ✓
F1 improving       ✓
→ Everything working perfectly
```

### Scenario 2: Loss Stuck, Gradients Fine
```
Loss plateauing         ⚠️
Gradients still large   ✓
F1 not improving        ⚠️
→ May need: Different learning rate or different architecture
```

### Scenario 3: Loss Stuck, Gradients Dying
```
Loss plateauing        ⚠️
Gradients → 0          ⚠️
F1 not improving       ⚠️
→ Vanishing gradient problem
→ Solutions:
   - Reduce learning rate
   - Add batch normalization
   - Use residual connections
   - Check initialization
```

### Scenario 4: Loss Erratic, Gradients Massive
```
Loss jumps around     ⚠️
gradient_norm spikes  ⚠️
F1 erratic            ⚠️
→ Usually means:
   - Gradient clipping is active (set to 1.0)
   - Loss is numerically unstable
→ Check: Class imbalance, data normalization
```

---

## TensorBoard Navigation

1. In TensorBoard, go to **SCALARS** tab
2. Look for curves starting with `gradients/` and `train/gradient_norm`
3. Watch them alongside `train/loss` for correlation

### Pro Tip: Create Custom Dashboard
```
In TensorBoard:
1. Click "Create new dashboard"
2. Add:
   - train/loss
   - train/gradient_norm
   - gradients/input_linear
   - gradients/output_head
   - train/f1
3. Save as "GradientHealth" dashboard
4. View all together on one page
```

---

## Gradient Logging in Your Pipeline

```python
# Automatically logged during training via hook:

def on_after_backward(self):
    # Called after loss.backward()
    # Computes gradients for:
    # 1. All parameters combined (train/gradient_norm)
    # 2. Key layers (gradients/input_linear, etc.)
```

**When?** Every training step (10 step batches)  
**Where?** TensorBoard `train/` and `gradients/` namespace  
**Cost?** Negligible (~0.1% overhead)

---

## Quick Reference: What Numbers Mean

| Gradient Norm | Status | Interpretation |
|---|---|---|
| > 1.0 | Clipping Active | Loss landscape is steep; gradient clipping preventing explosion |
| 0.1 - 1.0 | Healthy | Normal range for stable training |
| 0.01 - 0.1 | Slowing | Learning is slowing down; may be near optimum |
| < 0.001 | Vanishing | Gradients essentially zero; training stalled |
| Growing over time | Alert | Possible divergence; usually caught by clipping |

---

## Files Updated

✅ `lightning_modules.py`
- Added `on_after_backward()` hook
- Logs 5 gradient metrics per training step
- Key layers monitored: input, transformer layer 0 & 5, output head

---

## Summary

You now have **8 total metrics in train/gradient space**:

```
TensorBoard Structure:
├── train/
│   ├── loss
│   ├── accuracy
│   ├── f1
│   ├── precision
│   ├── recall
│   └── gradient_norm              ← NEW
├── gradients/                      ← NEW
│   ├── input_linear
│   ├── transformer_layer0
│   ├── transformer_layer5
│   └── output_head
└── val/
    └── [metrics]
```

**This allows you to:**
- ✓ Catch training instabilities early
- ✓ Distinguish between loss plateau vs. vanishing gradients
- ✓ Debug if model stops learning
- ✓ Verify all layers are training (not stuck)

---

**Status:** ✅ Gradient logging active and ready  
**Start training:** `python lightning_train.py`

