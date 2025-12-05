# AI Models Summary

## Quick Answer

**Yes, this project contains manually trained hand gesture recognition models.**

**No, the emotion recognition model is NOT manually trained (it uses pre-trained models).**

---

## Details

### ✅ Hand Gesture Recognition - MANUALLY TRAINED

- **Models**: ResNet50 (47% accuracy) and MobileNetV2 (76.5% accuracy)
- **Dataset**: 22,890 images across 5 gesture classes
- **Training Script**: `backend/scripts/train_gesture_model.py`
- **Classes**: left_swipe, right_swipe, stop, thumbs_down, thumbs_up

### ❌ Emotion Recognition - PRE-TRAINED

- **Model**: FER+ ONNX (from ONNX Model Zoo)
- **No custom training**: Uses downloaded pre-trained weights
- **Classes**: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt

---

For full technical details, see [AI_MODELS_ANALYSIS.md](./AI_MODELS_ANALYSIS.md)
