# BFRB Classification Model - Complete Code Breakdown

## Overview
This notebook implements a deep learning solution for classifying Body-Focused Repetitive Behaviors (BFRBs) using multi-sensor data from wrist-worn Helios devices. The solution uses an ensemble of neural networks with sophisticated feature engineering and multi-modal sensor fusion.

## 1. Setup and Configuration

### 1.1 Imports and Dependencies
```python
# Core libraries for data processing and ML
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import warnings

# Scikit-learn for preprocessing and validation
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow/Keras for deep learning
from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise
)

# Additional libraries
import polars as pl  # High-performance DataFrame library
from scipy.spatial.transform import Rotation as R  # 3D rotation handling
```

### 1.2 Reproducibility Setup
```python
def seed_everything(seed):
    """Ensures reproducible results across all random number generators"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

### 1.3 Configuration Parameters
```python
TRAIN = False                     # Set to True for training mode
RAW_DIR = Path("data")           # Raw data directory
PRETRAINED_DIR = Path("OG-model") # Pre-trained models directory
EXPORT_DIR = Path("new-model")    # Output directory for new models
BATCH_SIZE = 64                  # Training batch size
PAD_PERCENTILE = 95              # Sequence padding percentile
LR_INIT = 5e-4                   # Initial learning rate
WD = 3e-3                        # Weight decay for regularization
MIXUP_ALPHA = 0.4                # MixUp augmentation parameter
EPOCHS = 160                     # Maximum training epochs
PATIENCE = 40                    # Early stopping patience
```

## 2. Neural Network Building Blocks

### 2.1 Tensor Manipulation Functions
```python
def time_sum(x):
    """Sums tensor along time axis for attention mechanism"""
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    """Removes last dimension from tensor"""
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    """Adds dimension to last axis of tensor"""
    return tf.expand_dims(x, axis=-1)
```

### 2.2 Squeeze-and-Excitation (SE) Block
```python
def se_block(x, reduction=8):
    """
    Implements Squeeze-and-Excitation mechanism for channel attention
    - Squeezes spatial information using Global Average Pooling
    - Excites channels through fully connected layers
    - Applies sigmoid gating to original features
    """
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)           # Squeeze: Global context
    se = Dense(ch // reduction, activation='relu')(se)  # Excitation: Dimensionality reduction
    se = Dense(ch, activation='sigmoid')(se)    # Excitation: Channel weights
    se = Reshape((1, ch))(se)                  # Reshape for broadcasting
    return Multiply()([x, se])                 # Apply channel attention
```

### 2.3 Residual CNN Block with SE
```python
def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    """
    Residual CNN block with Squeeze-and-Excitation attention
    - Two Conv1D layers with batch normalization
    - SE block for channel attention
    - Skip connection for gradient flow
    - Max pooling and dropout for regularization
    """
    shortcut = x
    
    # Two convolutional layers
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    # Apply squeeze-and-excitation
    x = se_block(x)
    
    # Skip connection with dimension matching
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = add([x, shortcut])        # Residual connection
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x
```

### 2.4 Attention Layer
```python
def attention_layer(inputs):
    """
    Implements temporal attention mechanism
    - Computes attention scores for each time step
    - Applies softmax normalization
    - Weights and sums temporal features
    """
    score = Dense(1, activation='tanh')(inputs)    # Attention scores
    score = Lambda(squeeze_last_axis)(score)       # Remove last dimension
    weights = Activation('softmax')(score)         # Normalize attention weights
    weights = Lambda(expand_last_axis)(weights)    # Add dimension back
    context = Multiply()([inputs, weights])        # Apply attention weights
    context = Lambda(time_sum)(context)            # Sum weighted features
    return context
```

## 3. Data Preprocessing and Feature Engineering

### 3.1 Sequence Preprocessing
```python
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    """
    Preprocesses time series sequences:
    - Forward and backward fill for missing values
    - Zero-fill for remaining NaN values
    - Standard scaling using pre-fitted scaler
    """
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')
```

### 3.2 MixUp Data Augmentation
```python
class MixupGenerator(Sequence):
    """
    Implements MixUp data augmentation for regularization
    - Linearly combines pairs of training examples
    - Mixes both features and labels
    - Helps improve generalization and calibration
    """
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
    
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        
        # MixUp implementation
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
```

### 3.3 Physics-Based Feature Engineering

#### 3.3.1 Gravity Removal from Accelerometer
```python
def remove_gravity_from_acc(acc_data, rot_data):
    """
    Removes gravitational component from accelerometer readings
    - Uses quaternion rotation data to determine device orientation
    - Transforms world-frame gravity to sensor frame
    - Subtracts gravity to get linear acceleration
    """
    # Extract acceleration and quaternion data
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    
    gravity_world = np.array([0, 0, 9.81])  # Gravity vector in world frame
    linear_accel = np.zeros_like(acc_values)
    
    for i in range(acc_values.shape[0]):
        if not (np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0))):
            try:
                rotation = R.from_quat(quat_values[i])
                # Transform gravity to sensor frame and subtract
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
            except ValueError:
                linear_accel[i, :] = acc_values[i, :]  # Fallback to raw acceleration
    
    return linear_accel
```

#### 3.3.2 Angular Velocity from Quaternions
```python
def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    """
    Derives angular velocity from quaternion time series
    - Calculates relative rotation between consecutive frames
    - Converts to rotation vector and scales by time delta
    - Provides additional motion information beyond raw IMU
    """
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_vel = np.zeros((quat_values.shape[0], 3))
    
    for i in range(quat_values.shape[0] - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]
        
        if not (np.all(np.isnan(q_t)) or np.all(np.isnan(q_t_plus_dt))):
            try:
                rot_t = R.from_quat(q_t)
                rot_t_plus_dt = R.from_quat(q_t_plus_dt)
                
                # Calculate relative rotation and convert to angular velocity
                delta_rot = rot_t.inv() * rot_t_plus_dt
                angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
            except ValueError:
                pass  # Keep zero if quaternion is invalid
                
    return angular_vel
```

#### 3.3.3 Angular Distance Calculation
```python
def calculate_angular_distance(rot_data):
    """
    Calculates angular distance between consecutive quaternions
    - Measures amount of rotation between time steps
    - Useful for detecting rapid orientation changes
    - Complements angular velocity information
    """
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros(quat_values.shape[0])
    
    for i in range(quat_values.shape[0] - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        
        if not (np.all(np.isnan(q1)) or np.all(np.isnan(q2))):
            try:
                r1, r2 = R.from_quat(q1), R.from_quat(q2)
                relative_rotation = r1.inv() * r2
                angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
            except ValueError:
                pass
                
    return angular_dist
```

## 4. Model Architecture

### 4.1 Two-Branch Neural Network
```python
def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    """
    Builds a two-branch neural network for multi-modal sensor fusion:
    
    Branch 1 (IMU): Deep processing for inertial data
    - Residual CNN blocks with SE attention
    - Captures temporal patterns in motion data
    
    Branch 2 (ToF/Thermal): Lighter processing for environmental data
    - Simple CNN layers
    - Processes distance and temperature information
    
    Fusion: Combines both branches with attention mechanism
    """
    
    # Input layer for combined IMU + ToF/Thermal data
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    
    # Split input into IMU and ToF branches
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU Branch: Deep residual processing
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # ToF/Thermal Branch: Lighter processing
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)

    # Merge branches
    merged = Concatenate()([x1, x2])

    # Multi-path temporal processing
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)  # Noise regularization
    xc = Dense(16, activation='elu')(xc)
    
    # Combine all paths
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    
    # Apply temporal attention
    x = attention_layer(x)

    # Classification head with regularization
    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    # Output layer
    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)
```

## 5. Inference Pipeline

### 5.1 Model Loading and Ensemble Setup
```python
# Load preprocessing artifacts
final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

# Define custom objects for model loading
custom_objs = {
    'time_sum': time_sum, 
    'squeeze_last_axis': squeeze_last_axis, 
    'expand_last_axis': expand_last_axis,
    'se_block': se_block, 
    'residual_se_cnn_block': residual_se_cnn_block, 
    'attention_layer': attention_layer,
}

# Load ensemble of 20 models (2 different architectures × 10 folds each)
models = []
for fold in range(10):
    # Load first architecture
    model_path = f"new-model/D-111_{fold}.h5"
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models.append(model)

for fold in range(10):
    # Load second architecture
    model_path = f"new-model/v0629_{fold}.h5"
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models.append(model)
```

### 5.2 Prediction Function
```python
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Main prediction function for BFRB classification
    Processes raw sensor data through complete feature engineering pipeline
    """
    
    # Convert to pandas for processing
    df_seq = sequence.to_pandas()
    
    # Apply physics-based feature engineering
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'] = linear_accel[:, 0]
    df_seq['linear_acc_y'] = linear_accel[:, 1] 
    df_seq['linear_acc_z'] = linear_accel[:, 2]
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + 
                                      df_seq['linear_acc_y']**2 + 
                                      df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
    
    # Calculate angular features
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'] = angular_vel[:, 0]
    df_seq['angular_vel_y'] = angular_vel[:, 1]
    df_seq['angular_vel_z'] = angular_vel[:, 2]
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)
    
    # Process ToF sensor data (5 sensors with 64 pixels each)
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = df_seq[pixel_cols].replace(-1, np.nan)  # Handle no-response values
        
        # Statistical features for each ToF sensor
        df_seq[f'tof_{i}_mean'] = tof_data.mean(axis=1)
        df_seq[f'tof_{i}_std'] = tof_data.std(axis=1)
        df_seq[f'tof_{i}_min'] = tof_data.min(axis=1)
        df_seq[f'tof_{i}_max'] = tof_data.max(axis=1)
    
    # Prepare input for neural network
    mat_unscaled = df_seq[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = scaler.transform(mat_unscaled)
    pad_input = pad_sequences([mat_scaled], maxlen=pad_len, 
                             padding='post', truncating='post', dtype='float32')
    
    # Ensemble prediction
    all_preds = [model.predict(pad_input, verbose=0)[0] for model in models]
    avg_pred = np.mean(all_preds, axis=0)
    
    # Return predicted gesture class
    return str(gesture_classes[avg_pred.argmax()])
```

## 6. Evaluation and Deployment

### 6.1 Kaggle Evaluation Server
```python
import kaggle_evaluation.cmi_inference_server

# Initialize inference server with prediction function
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

# Run appropriate mode based on environment
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()  # Competition mode
else:
    # Local testing mode
    inference_server.run_local_gateway(
        data_paths=('/data/test.csv', '/data/test_demographics.csv')
    )
```

## 7. Key Technical Features

### 7.1 Multi-Modal Architecture
- **Two-Branch Design**: Separate processing for IMU and ToF/Thermal data
- **Attention Mechanisms**: SE blocks for channel attention, temporal attention for sequence modeling
- **Ensemble Learning**: 20 models for robust predictions

### 7.2 Advanced Feature Engineering
- **Physics-Based Features**: Gravity removal, angular velocity, angular distance
- **Statistical Aggregation**: Mean, std, min, max for ToF sensor grids
- **Temporal Derivatives**: Jerk (acceleration derivative) for motion analysis

### 7.3 Regularization Techniques
- **Dropout**: Multiple dropout layers with varying rates
- **Weight Decay**: L2 regularization throughout the network
- **Batch Normalization**: Stabilizes training and improves convergence
- **MixUp**: Data augmentation for better generalization
- **Gaussian Noise**: Additional regularization in temporal processing

### 7.4 Robustness Features
- **Missing Data Handling**: Forward/backward fill, zero-fill strategies
- **Invalid Sensor Data**: Graceful handling of sensor communication failures
- **Cross-Validation**: Subject-wise splitting to prevent data leakage

## 8. Performance Characteristics

### 8.1 Model Complexity
- **Parameters**: Large ensemble with 20 models
- **Memory Usage**: Significant due to ToF sensor data (320 features per timestep)
- **Inference Time**: Optimized for 30-minute prediction constraint

### 8.2 Validation Strategy
- **Cross-Validation**: Stratified Group K-Fold (subject-wise)
- **Metrics**: Macro F1 score combining binary and multi-class classification
- **OOF Score**: 0.8121 (Out-of-Fold validation)

This architecture represents a sophisticated approach to multi-modal sensor fusion for human activity recognition, combining domain-specific feature engineering with modern deep learning techniques for robust BFRB classification. 