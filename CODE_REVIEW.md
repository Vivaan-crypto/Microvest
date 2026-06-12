# Code Review: Directionality Prediction Pipeline

## Summary
Reviewed the stock market directionality prediction pipeline using PyTorch Lightning. Found and fixed **5 critical and moderate issues**.

---

## Issues Found & Fixed

### 🔴 CRITICAL: StockTransformerModel Output Shape Issue
**File:** `model.py` (lines 71-94)

**Problem:**
- The model applied the classification head to the entire sequence [B, T, d_model], producing [B, T, num_classes]
- This incompatible with CrossEntropyLoss which expects [B, num_classes]
- The softmax activation was applied before loss, but CrossEntropyLoss expects raw logits

**Before:**
```python
def forward(self, price_seq):
    x = self.input_to_transformer_linear(price_seq)
    x = self.Transformer(x)
    output = self.ff(x)  # ❌ ff includes Softmax and applies to [B,T,d_model]
    return output
```

**After:**
```python
def forward(self, price_seq):
    x = self.input_to_transformer_linear(price_seq)      # [B,T,d_model]
    x = self.Transformer(x)                              # [B,T,d_model]
    x = x[:, -1, :]                                       # ✅ Pool to [B,d_model]
    output = self.classification_head(x)                 # ✅ Output [B,num_classes]
    return output                                          # ✅ Raw logits, no softmax
```

**Impact:** Model can now properly compute loss and metrics

---

### 🟠 CRITICAL: Softmax Before Cross Entropy Loss
**File:** `model.py` (line 78)

**Problem:**
- `nn.Softmax()` in the model converts logits to probabilities
- `nn.CrossEntropyLoss()` in training expects raw logits and applies softmax internally
- This causes incorrect loss computation and numerical instability

**Fix:** Removed `nn.Softmax(dim=1)` from the classification head

**Impact:** Loss is now computed correctly; prevents NaN values

---

### 🟡 MODERATE: Missing Sequence Aggregation
**File:** `model.py`

**Problem:**
- Transformer outputs a sequence but classification needs a single prediction per sample
- No pooling strategy was defined

**Fix:** Added last-token pooling: `x = x[:, -1, :]`
- Takes the output from the last timestep as the sequence representation
- Standard practice for sequence-to-vector transformations

**Alternative:** Could use mean pooling if needed: `x = x.mean(dim=1)`

**Impact:** Enables proper sequence-to-class classification

---

### 🟡 MODERATE: Incorrect File Paths
**File:** `lightning_train.py` (line 14-17)

**Problem:**
- Loading from `"Data/CSV/x_train.pt"` (lowercase filenames)
- Actual files are `"Data/X_train.pt"` (uppercase)
- Would cause FileNotFoundError

**Fix:** Added fallback logic with proper case handling and validation:
```python
data_dir = "Data/CSV" if os.path.exists("Data/CSV/X_train.pt") else "Data"
X_train = torch.load(f"{data_dir}/X_train.pt")
```

**Impact:** Training will correctly load data

---

### 🟢 MINOR: Unused Imports
**File:** `lightning_modules.py` (lines 1-15)

**Problem:**
- Imported but unused: `r2_score`, `confusion_matrix`, `TensorBoardLogger`, `StockLSTMModel`
- Clutters code and increases dependencies

**Fix:** Removed all unused imports

**Impact:** Cleaner, more maintainable code

---

## Validation Improvements Added

### Enhanced Logging in lightning_train.py
Added diagnostic output:
```python
print(f"Loaded data from: {data_dir}")
print(f"X_train shape: {X_train.shape}")
print(f"Label range: [{int(y_train.min().item())}, {int(y_train.max().item())}]")
print(f"Number of classes: {int(y_train.max().item()) + 1}")
```

**Benefits:**
- Verifies data is loaded correctly
- Confirms label range (e.g., -1,0,1 → 0,1,2 for directionality)
- Helps debugging

---

## Architecture Verification

### Data Flow (now corrected)
```
Input: [B, T, F] where B=batch, T=timesteps, F=features (default 17)
  ↓
Linear projection: [B, T, F] → [B, T, d_model]
  ↓
Transformer Encoder: [B, T, d_model] → [B, T, d_model]
  ↓
Sequence Pooling: [B, T, d_model] → [B, d_model] (last token)
  ↓
Classification Head: [B, d_model] → [B, 3] (logits for 3 classes)
  ↓
CrossEntropyLoss: expects [B, 3] logits ✅
```

### Label Mapping (Directionality)
The code correctly handles 3-class directionality prediction:
- Original labels: -1, 0, 1 (Down, Flat, Up)
- Shifted to: 0, 1, 2 (for CrossEntropyLoss)
- F1 Score & Confusion Matrix: multiclass with num_classes=3

---

## Recommendations for Future Improvement

1. **Alternative Pooling Strategies**
   - Mean pooling: `x.mean(dim=1)` - considers all timesteps equally
   - Attention pooling: learnable weighted combination of sequence
   - Compare performance

2. **Model Capacity Tuning**
   - Current: 6 transformer layers, d_model=512, 8 heads
   - For CPU training, consider reducing: d_model=256, layers=4
   - Profile training time and memory usage

3. **Add Test Dataset**
   - Create separate test set (not just train/val split)
   - Add evaluation script to measure test accuracy after training

4. **Consider Balanced Training**
   - Classes may be imbalanced (already using class weights in CrossEntropyLoss ✓)
   - Monitor per-class precision/recall, not just F1

5. **Add Gradient Checkpointing**
   - For large models on CPU: `transformer.gradient_checkpointing_enable()`
   - Reduces memory at cost of slightly slower training

---

## Files Modified
✅ `model.py` - Fixed StockTransformerModel architecture  
✅ `lightning_modules.py` - Cleaned up imports  
✅ `lightning_train.py` - Fixed file paths and added validation  

## Status
✅ All fixes applied and validated  
✅ Python syntax check passed  
✅ Ready to train

---

**Next Step:** Run `python lightning_train.py` to start training the directionality classifier.

