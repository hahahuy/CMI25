# Solution Guide: Fixing Feature Dimension Mismatch

## Problem Summary
- **Error**: `expected shape=(None, 127, 38), found shape=(1, 127, 361)`
- **Root Cause**: Training and inference use different feature sets
- **Impact**: Model fails during evaluation server submission

## Solution Options

### Option 1: Retrain with Consistent Features (Recommended)

**Steps:**

1. **Set TRAIN = True** in the training notebook
2. **Use the fixed training pipeline** that ensures consistent features
3. **Train new models** with exactly 41 features:
   - 7 base IMU features
   - 9 physics-derived features  
   - 5 thermal features
   - 20 ToF statistical features (no raw pixels)

**Expected Results:**
- Model input shape: `(None, 127, 41)`
- Consistent training/inference pipeline
- No dimension mismatch errors

### Option 2: Fix Existing Model (Quick Fix)

If you want to keep existing models, you need to match the 38-feature format they expect.

**Analysis of 38 features:**
The pretrained model expects 38 features, which suggests it was trained without some features. 
You need to identify exactly which 38 features were used during original training.

## Implementation

### For Option 1 (Retrain):

```python
# In training notebook, ensure this exact feature set:
base_features = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
physics_features = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 
                   'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 
                   'angular_distance']
thermal_features = [f'thm_{i}' for i in range(1, 6)]
tof_stat_features = [f'tof_{i}_{stat}' for i in range(1, 6) for stat in ['mean', 'std', 'min', 'max']]

# Use ONLY these 41 features
final_feature_cols = base_features + physics_features + thermal_features + tof_stat_features
```

### For Option 2 (Match existing):

```python
# You need to determine which 38 features the model expects
# Check the saved feature_cols.npy file:
import numpy as np
existing_features = np.load("OG-model/feature_cols.npy", allow_pickle=True)
print(f"Model expects {len(existing_features)} features:")
print(existing_features)
```

## Key Changes Made

### 1. Fixed Training Pipeline
- **Consistent feature engineering** between training and inference
- **Explicit feature set definition** 
- **No raw ToF pixels** - use statistical features only
- **Proper dimension calculation** for model architecture

### 2. Fixed Inference Function
- **Same feature engineering** as training
- **Handle missing ToF data** gracefully
- **Use only saved feature columns**
- **Debug prints** for troubleshooting

### 3. Model Architecture Alignment
```python
imu_dim = 16  # base(7) + physics(9)
tof_dim = 25  # thermal(5) + tof_stats(20)
total = 41    # Expected input features
```

## Verification Steps

1. **Check feature consistency**:
   ```bash
   python3 verify_features.py
   ```

2. **Verify model input shape**:
   ```python
   print(f"Model input shape: {model.input_shape}")
   print(f"Data shape: {X.shape}")
   ```

3. **Test prediction function**:
   ```python
   # Should not raise dimension errors
   result = predict(test_sequence, test_demographics)
   ```

## Expected Performance
- **Model Architecture**: Two-branch CNN + LSTM/GRU + Attention
- **Features**: 41 statistical + physics features (no raw pixels)
- **Performance**: ~0.81-0.82 on validation
- **Efficiency**: Faster inference without 320 pixel features

## Troubleshooting

If you still get dimension errors:
1. Check `feature_cols.npy` content
2. Verify feature engineering consistency
3. Print actual vs expected shapes
4. Use debug prints in prediction function

## Recommendation

**Use Option 1 (Retrain)** because:
- Ensures complete consistency
- Better performance with proper features
- Cleaner, more maintainable code
- Future-proof solution