# Body-Focused Repetitive Behaviors (BFRBs) Classification Challenge

## Project Overview

This project implements a machine learning solution for classifying body-focused repetitive behaviors (BFRBs) using multi-sensor data from a wrist-worn Helios device. The goal is to accurately identify 8 BFRB-like gestures and 10 non-BFRB-like gestures from sensor recordings.

## Key Components

### 1. Sensor Data Types
- **IMU (Inertial Measurement Unit)**: Accelerometer, gyroscope, and magnetometer data
  - `acc_x/y/z`: Linear acceleration along three axes (m/s²)
  - `rot_w/x/y/z`: 3D orientation data (quaternions)
- **5x Thermopile Sensors (MLX90632)**: Non-contact temperature sensors
  - `thm_1` through `thm_5`: Temperature measurements in degrees Celsius
- **5x Time-of-Flight Sensors (VL53L7CX)**: Distance measurement sensors
  - `tof_[1-5]_v[0-63]`: 64-pixel 8x8 grid distance measurements per sensor
  - Values range 0-254 (uncalibrated), -1 indicates no response

### 2. Data Structure
- **Sequences**: Each sequence contains one Transition, one Pause, and one Gesture
- **Subjects**: Multiple participants with demographic information
- **Training Data**: 8 BFRB gestures + 10 non-BFRB gestures
- **Test Data**: ~3,500 sequences (50% IMU-only, 50% full sensor data)

### 3. Target Variables
- **Primary**: `gesture` - Classification of 18 different gesture types
- **Secondary**: `sequence_type` - Binary classification (target vs non-target)

### 4. Evaluation Metrics
- **Macro F1 Score**: Average of two components:
  - Binary F1 on target vs non-target classification
  - Macro F1 on gesture classification (non-target sequences collapsed)
- **Submission**: Must use provided evaluation API with 30-minute prediction limit

## Key Challenges

### 1. Multi-Modal Sensor Fusion
- Combining heterogeneous sensor data (IMU, temperature, distance)
- Handling different data scales and temporal characteristics
- Managing sensor communication failures and missing data

### 2. Data Complexity
- Large feature space: 5 ToF sensors × 64 pixels = 320 distance features
- Temporal dependencies across sequence phases
- Subject-specific variations in movement patterns

### 3. Real-Time Constraints
- Inference must complete within 30 minutes per sequence
- Memory and computational efficiency requirements
- Sequential processing (one sequence at a time)

### 4. Class Imbalance
- 8 BFRB vs 10 non-BFRB gesture classes
- Potential imbalance within gesture categories
- Subject-specific gesture frequency variations

## Data Features

### Sensor Features
- **IMU Features**: 6 features (3 acceleration + 3 orientation)
- **Thermopile Features**: 5 temperature sensors
- **Time-of-Flight Features**: 320 distance measurements (5 sensors × 64 pixels)

### Metadata Features
- **Sequence Information**: sequence_id, sequence_counter, phase
- **Subject Demographics**: age, sex, handedness, height, arm measurements
- **Behavioral Context**: orientation, behavior, sequence_type

### Derived Features (To Be Engineered)
- **Statistical Features**: mean, std, min, max, percentiles
- **Frequency Domain**: FFT coefficients, power spectral density
- **Temporal Features**: velocity, acceleration derivatives, jerk
- **Cross-Sensor**: correlation matrices, interaction features

## Recommended Methods

### 1. Deep Learning Architecture
```
Multi-Modal Fusion Network:
├── IMU Encoder (CNN + LSTM)
├── Thermopile Encoder (LSTM)
├── ToF Encoder (CNN for spatial features)
├── Attention Fusion Layer
├── Temporal Processing (Transformer/LSTM)
└── Classification Head (Multi-task)
```

### 2. Feature Engineering Strategy
- **Temporal Segmentation**: Sliding windows with overlap
- **Sensor-Specific Processing**: 
  - IMU: Velocity and acceleration derivatives
  - ToF: Spatial convolution for grid data
  - Thermopile: Temperature gradients and patterns
- **Subject Normalization**: Using demographic information

### 3. Training Strategy
- **Subject-Wise Cross-Validation**: Prevent data leakage
- **Weighted Loss Functions**: Handle class imbalance
- **Data Augmentation**: Noise injection, time warping
- **Ensemble Methods**: Combine multiple model architectures

## Key Considerations

- **Memory Efficiency**: Large ToF sensor data (5 sensors × 64 pixels)
- **Real-Time Constraints**: 30-minute prediction limit per sequence
- **Robustness**: Handle missing sensors and communication failures
- **Interpretability**: Understand which sensors contribute most to classification
- **Scalability**: Efficient processing for ~3,500 test sequences

### **ℹ️INFO**
* First, we would like to thank our participants for sharing their excellent baselines.
    * [Two‑Branch Human‑Activity‑Recognition Pipeline (IMU + Thermopile/TOF  + SE‑CNN + BiLSTM + Attentio](https://www.kaggle.com/code/vonmainstein/imu-tof)

### **ℹ️IMU+THM/TOF Great Related works(Published order)**
* Thanks for sharing your implementation of the new feature.
    * LB.75 [CMI25 | IMU+THM/TOF |TF BiLSTM+GRU+Attention|LB.75](https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-bilstm-gru-attention-lb-75)
    * LB.76 [IMU Signal Processing Optimization](https://www.kaggle.com/code/rktqwe/lb-0-76-imu-thm-tof-tf-bilstm-gru-attention)
    * LB.77 [Gravity Component Removal from Accelerometer Data](https://www.kaggle.com/code/rktqwe/lb-0-77-linear-accel-tf-bilstm-gru-attention)
    * LB.78 [New Feature: Angular Velocity from Quaternion Derivatives](https://www.kaggle.com/code/nksusth/lb-0-78-quaternions-tf-bilstm-gru-attention)
    * LB.80 [IMU × TOF/THM Fusion with Physics‑FE & MixUp|LB0.8](https://www.kaggle.com/code/pepushi/imu-tof-thm-fusion-with-physics-fe-mixup-lb0-8)

---

### **ℹ️[LB.80 2025/06/28]MyUpdate**
* In previous notebooks that used "IMU+TOF" as features, the validation strategy was TTS, which raised concerns about its robustness.
For this reason, we will release an implementation that simply blends five models and a model with a high ValidationScore.

```
predictions = []
for model in models:
    idx = int(model.predict(pad_input, verbose=0).argmax(1)[0])
    predictions.append(idx)

idx = max(set(predictions), key=predictions.count)
return str(gesture_classes[idx])
```

* The validationScore for the local training  model included in this notebook is
    * [ModelWeight](https://www.kaggle.com/datasets/hideyukizushi/20250627-cmi-b-102-b-105)

```
0.891134700273056
0.8912659261884439
0.8914825129445727
0.8915471835009202
0.8922128108549205
```

### **ℹ️[LB.82 2025/07/09]MyUpdate**
* blend use SGKF split model(The model is attached in a notebook and is publicly available.)
    * [ModelWeight](https://www.kaggle.com/datasets/hideyukizushi/cmi-d-111)

```
# OOF(CMI Metrics)
0.8120966859982701
```