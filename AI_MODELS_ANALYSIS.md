# AI Models Analysis Report

## Summary

This repository contains **one manually trained hand gesture recognition model** and **one pre-trained emotion recognition model**.

---

## 1. Hand Gesture Recognition Model ✅ MANUALLY TRAINED

### Overview
The project includes **manually trained hand gesture recognition models** using Transfer Learning with two different architectures.

### Model Details

#### Architecture
- **Model 1: ResNet50** - Deep residual network
  - Location: `backend/models/gesture_resnet50.keras`
  - Model Size: **98 MB**
  - Validation Accuracy: **47.07%**
  - Top-2 Accuracy: **72.83%**
  - Input Size: 160x160x3
  
- **Model 2: MobileNetV2** - Lightweight efficient network (BEST PERFORMING)
  - Location: `backend/models/gesture_mobilenetv2.keras`
  - Model Size: **12 MB**
  - Validation Accuracy: **76.53%**
  - Top-2 Accuracy: **92.97%**
  - Input Size: 160x160x3

#### Gesture Classes
The models can recognize 5 hand gestures:
1. `left_swipe` 👈 - Swipe Left
2. `right_swipe` 👉 - Swipe Right  
3. `stop` ✋ - Stop
4. `thumbs_down` 👎 - Thumbs Down
5. `thumbs_up` 👍 - Thumbs Up

### Training Process

#### Training Script
- **Location**: `backend/scripts/train_gesture_model.py`
- **Lines**: 375 lines of comprehensive training code
- **Method**: Transfer Learning with frozen pre-trained weights from ImageNet

#### Dataset
- **Location**: `backend/datasets/dataset_hand_gesture/`
- **Size**: 
  - Training images: **19,890 PNG images** (663 folders, ~30 images per folder)
  - Validation images: **3,000 PNG images** (100 folders, ~30 images per folder)
  - Total: **22,890 images** across 5 gesture classes
- **Structure**:
  - Training set: `train.csv` lists 663 folders, each containing multiple PNG images
  - Validation set: `val.csv` lists 100 folders, each containing multiple PNG images
  - Each row in CSV represents a folder with multiple image frames from a video sequence
- **Format**: CSV with format `folder_name;class_name;class_id`

#### Training Configuration
```python
IMG_SIZE = (160, 160)          # Optimized image size
BATCH_SIZE = 64                # Batch size for training
EPOCHS = 12                     # Maximum epochs
LEARNING_RATE = 0.0001         # Adam optimizer learning rate
```

#### Data Augmentation
- Rotation: ±15 degrees
- Width/Height shift: ±15%
- Horizontal flip
- Zoom: ±15%
- Rescaling: 1/255

#### Training Features
1. **Early Stopping**: Monitors validation loss with patience of 5 epochs
2. **Learning Rate Reduction**: Reduces LR by 0.5x when validation loss plateaus
3. **Best Weights Restoration**: Automatically restores best model weights
4. **Metrics Tracked**: 
   - Categorical Cross-Entropy Loss
   - Accuracy
   - Top-2 Accuracy

#### Model Architecture
Both models use the same pattern:
```
Input (160x160x3)
  ↓
Pre-trained Base Model (frozen)
  ↓
Global Average Pooling
  ↓
Dense Layer(s) with ReLU
  ↓
Dropout Layer(s)
  ↓
Output Dense (5 classes, softmax)
```

### Service Implementation

#### Classifier Service
- **Location**: `backend/services/gesture_classifier.py`
- **Lines**: 319 lines
- **Features**:
  - Lazy loading of TensorFlow/Keras
  - Enhanced preprocessing with CLAHE and denoising
  - Support for file, bytes, and numpy array inputs
  - Ensemble prediction capability (uses MobileNetV2 as primary)
  - Confidence scoring with softmax
  - Global singleton pattern for efficiency

#### API Endpoint
```http
POST /api/v1/classify/gesture
Content-Type: multipart/form-data

image: <image file>
```

**Response:**
```json
{
  "gesture": "thumbs_up",
  "confidence": 0.85,
  "emoji": "👍",
  "display_name": "Thumbs Up",
  "all_predictions": {...},
  "top_3": [...]
}
```

### Metadata Files
- `backend/models/gesture_classes.txt` - Class labels
- `backend/models/gesture_resnet50_metadata.json` - ResNet50 metrics
- `backend/models/gesture_mobilenetv2_metadata.json` - MobileNetV2 metrics

---

## 2. Emotion Recognition Model ❌ NOT MANUALLY TRAINED

### Overview
The emotion recognition system uses **pre-trained models** from external sources, not manually trained models.

### Model Details

#### Primary Model: FER+ (ONNX)
- **Location**: `backend/models/emotion-ferplus-8.onnx`
- **Model Size**: **182 KB**
- **Source**: ONNX Model Zoo (pre-trained)
- **Model URL**: https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
- **Type**: Pre-trained convolutional neural network
- **Input**: 64x64 grayscale images
- **Output**: 8 emotion classes

#### Emotion Classes
1. `neutral`
2. `happiness`
3. `surprise`
4. `sadness`
5. `anger`
6. `disgust`
7. `fear`
8. `contempt`

#### Secondary Option: DeepFace
- **Location**: `backend/services/emotion_deepface.py`
- **Type**: Library-based (uses pre-trained DeepFace models)
- **Backend**: OpenCV for face detection
- **Not manually trained**: Uses off-the-shelf DeepFace library

### Service Implementation

#### ONNX Service
- **Location**: `backend/services/emotion_local_onnx.py`
- **Features**:
  - Automatic model download if not present
  - Haar Cascade for face detection
  - Multi-face support
  - Aggregated emotion across all faces

#### DeepFace Service
- **Location**: `backend/services/emotion_deepface.py`
- **Features**:
  - Integration with DeepFace library
  - Multiple detector backend support (opencv, ssd, etc.)
  - TensorFlow/Keras compatibility handling

#### API Endpoint
```http
POST /api/v1/face/sentiment
Content-Type: multipart/form-data

image: <image file>
```

**Response:**
```json
{
  "dominant_emotion": "happiness",
  "face_count": 1,
  "details": [
    {
      "box": [x, y, w, h],
      "top_emotion": "happiness",
      "scores": {
        "happiness": 0.85,
        "neutral": 0.10,
        ...
      }
    }
  ]
}
```

### Training Status
- **No manual training script found** for emotion recognition
- **No custom dataset** for emotion training
- **Uses pre-trained models** downloaded from external sources
- The FER+ model is downloaded automatically if not present

---

## Conclusion

### Summary Table

| Model Type | Manually Trained? | Model Names | Accuracy | Location |
|------------|------------------|-------------|----------|----------|
| **Hand Gesture Recognition** | ✅ YES | ResNet50, MobileNetV2 | 47% / 76.5% | `backend/models/gesture_*.keras` |
| **Emotion Recognition** | ❌ NO | FER+ (ONNX), DeepFace | N/A (Pre-trained) | `backend/models/emotion-ferplus-8.onnx` |

### Key Findings

1. **Hand Gesture Recognition**: 
   - Fully manually trained using Transfer Learning
   - Custom dataset with 5 gesture classes
   - Complete training pipeline with data augmentation
   - Production-ready with API endpoint
   - MobileNetV2 achieves 76.5% validation accuracy

2. **Emotion Recognition**: 
   - Uses pre-trained models (FER+ from ONNX Model Zoo)
   - No manual training performed
   - Relies on external pre-trained weights
   - DeepFace library as alternative option

### Recommendations

If you want to manually train an emotion recognition model similar to the gesture model:

1. **Collect/Obtain Dataset**: 
   - FER2013, AffectNet, or custom dataset
   - Labeled facial expressions with emotion classes

2. **Create Training Script**: 
   - Similar structure to `train_gesture_model.py`
   - Use Transfer Learning (ResNet50, MobileNetV2, or VGGFace)
   - Implement data augmentation for faces

3. **Architecture Suggestions**:
   - Use same Transfer Learning approach
   - Consider face-specific pre-trained models (VGGFace, FaceNet)
   - Add face detection preprocessing step

4. **Expected Structure**:
   ```
   backend/
   ├── datasets/
   │   └── emotion_dataset/
   │       ├── train.csv
   │       ├── val.csv
   │       └── images/
   ├── scripts/
   │   └── train_emotion_model.py  # NEW
   └── models/
       ├── emotion_resnet50.keras  # NEW
       └── emotion_mobilenetv2.keras  # NEW
   ```

---

## Additional AI Models in Project

The project also contains training scripts for other predictive models (not vision-based):

1. **BMI/Body Fat Model**: `train_bmi_model.py`
2. **Bitcoin Price Prediction**: `train_bitcoin_model.py`
3. **Car Price Prediction**: `train_car_price.py`
4. **Avocado Price Prediction**: `train_avocado_model.py`
5. **London Crime Classification**: `train_london_crime_model.py`
6. **Cirrhosis Prediction**: `train_cirrhosis_model.py`
7. **Audio CTC Model**: `train_audio_ctc.py`

All of these are manually trained ML models for various prediction tasks.

---

**Report Generated**: 2025-12-05  
**Analysis Scope**: AI Vision Models (Emotion Recognition & Hand Gesture Recognition)
