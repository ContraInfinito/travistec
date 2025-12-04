"""
Gesture Classification Service

Loads pre-trained gesture recognition models and provides prediction functions.
Supports both ResNet50 and MobileNetV2 models for ensemble predictions.

Usage:
    from services.gesture_classifier import GestureClassifier
    
    classifier = GestureClassifier()
    result = classifier.predict_from_file("path/to/image.jpg")
    # Returns: {'gesture': 'thumbs_up', 'confidence': 0.95, 'all_predictions': {...}}
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2

# Lazy imports for faster startup
_tf = None
_keras = None


def _ensure_tensorflow():
    """Lazy load TensorFlow and Keras."""
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        import keras  # Use keras 3 directly, not tf.keras
        _tf = tf
        _keras = keras
    return _tf, _keras


class GestureClassifier:
    """Gesture recognition using transfer learning models."""
    
    # Gesture labels
    GESTURES = ['left_swipe', 'right_swipe', 'stop', 'thumbs_down', 'thumbs_up']
    
    # Emoji mapping for display
    GESTURE_EMOJIS = {
        'left_swipe': '👈',
        'right_swipe': '👉',
        'stop': '✋',
        'thumbs_down': '👎',
        'thumbs_up': '👍'
    }
    
    # Display names
    GESTURE_NAMES = {
        'left_swipe': 'Swipe Left',
        'right_swipe': 'Swipe Right',
        'stop': 'Stop',
        'thumbs_down': 'Thumbs Down',
        'thumbs_up': 'Thumbs Up'
    }
    
    # No confidence threshold - always return best prediction
    CONFIDENCE_THRESHOLD = 0.0  # Always return best prediction
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize gesture classifier.
        
        Args:
            models_dir: Directory containing trained models. If None, uses backend/models/
        """
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent / "models"
        
        self.models_dir = Path(models_dir)
        self.resnet_model = None
        self.mobilenet_model = None
        self.classes = []
        self.img_size = (160, 160)  # Match training configuration
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained gesture recognition models."""
        tf, keras = _ensure_tensorflow()
        
        # Suppress Keras verbose output during model loading
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Load class labels
        classes_path = self.models_dir / "gesture_classes.txt"
        if classes_path.exists():
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            print(f"[GestureClassifier] Loaded {len(self.classes)} gesture classes")
        else:
            # Fallback to default classes
            self.classes = self.GESTURES
            print(f"[GestureClassifier] Using default gesture classes")
        
        # Load ResNet50 model
        resnet_path = self.models_dir / "gesture_resnet50.keras"
        if resnet_path.exists():
            try:
                self.resnet_model = keras.models.load_model(str(resnet_path), compile=False)
                print(f"[GestureClassifier] ✅ Loaded ResNet50 model")
            except Exception as e:
                print(f"[GestureClassifier] ⚠️  Failed to load ResNet50: {str(e)[:100]}")
        else:
            print(f"[GestureClassifier] ⚠️  ResNet50 model not found")
        
        # Load MobileNetV2 model
        mobilenet_path = self.models_dir / "gesture_mobilenetv2.keras"
        if mobilenet_path.exists():
            try:
                self.mobilenet_model = keras.models.load_model(str(mobilenet_path), compile=False)
                print(f"[GestureClassifier] ✅ Loaded MobileNetV2 model")
            except Exception as e:
                print(f"[GestureClassifier] ⚠️  Failed to load MobileNetV2: {str(e)[:100]}")
        else:
            print(f"[GestureClassifier] ⚠️  MobileNetV2 model not found")
        
        if not self.resnet_model and not self.mobilenet_model:
            print("[GestureClassifier] ❌ No gesture models loaded!")
            print("[GestureClassifier] Run: python backend/scripts/train_gesture_model.py")
    
    def is_available(self) -> bool:
        """Check if at least one model is loaded."""
        return self.resnet_model is not None or self.mobilenet_model is not None
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing with CLAHE and denoising for better gesture distinction.
        
        Args:
            img: BGR image from OpenCV
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size with high-quality interpolation
        img_resized = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply Gaussian blur to reduce noise
        img_denoised = cv2.GaussianBlur(img_resized, (3, 3), 0)
        
        # Convert BGR to LAB for better color processing
        img_lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        
        # Convert back to BGR then to RGB
        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        
        # MobileNetV2 preprocessing: scale to [-1, 1]
        img_normalized = (img_rgb.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def _ensemble_predict(self, img_tensor: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Make prediction using MobileNetV2 (best performing model).
        
        Args:
            img_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Use ONLY MobileNetV2 (76% accuracy) - best model
        if self.mobilenet_model:
            pred = self.mobilenet_model.predict(img_tensor, verbose=0)[0]
        elif self.resnet_model:
            # Fallback to ResNet50 if MobileNetV2 not available
            pred = self.resnet_model.predict(img_tensor, verbose=0)[0]
        else:
            raise ValueError("No models available")
        
        # Apply softmax for better probability distribution
        exp_pred = np.exp(pred - np.max(pred))  # Numerical stability
        softmax_pred = exp_pred / np.sum(exp_pred)
        
        # Get predicted class
        pred_idx = np.argmax(softmax_pred)
        confidence = float(softmax_pred[pred_idx])
        predicted_class = self.classes[pred_idx]
        
        # Get all probabilities
        all_probs = {self.classes[i]: float(softmax_pred[i]) for i in range(len(self.classes))}
        
        return predicted_class, confidence, all_probs
    
    def predict_from_array(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Predict gesture from numpy array (OpenCV image).
        Always returns the best prediction available.
        
        Args:
            img: BGR image from OpenCV
            
        Returns:
            Dictionary with prediction results (only top prediction, no alternatives)
        """
        if not self.is_available():
            return {
                'error': 'no_models_loaded',
                'message': 'Gesture recognition models not available. Train models first.'
            }
        
        try:
            # Preprocess image
            img_tensor = self._preprocess_image(img)
            
            # Make prediction
            gesture, confidence, all_probs = self._ensemble_predict(img_tensor)
            
            # Always return the best prediction - no alternatives shown
            return {
                'gesture': gesture,
                'confidence': confidence,
                'emoji': self.GESTURE_EMOJIS.get(gesture, '🤚'),
                'display_name': self.GESTURE_NAMES.get(gesture, gesture),
                'all_predictions': {gesture: confidence},  # Only show top prediction
                'top_3': [
                    {
                        'gesture': gesture,
                        'confidence': confidence,
                        'emoji': self.GESTURE_EMOJIS.get(gesture, '🤚'),
                        'display_name': self.GESTURE_NAMES.get(gesture, gesture)
                    }
                ]
            }
        
        except Exception as e:
            return {
                'error': 'prediction_failed',
                'message': str(e)
            }
    
    def predict_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Predict gesture from image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        if not Path(file_path).exists():
            return {
                'error': 'file_not_found',
                'message': f'Image file not found: {file_path}'
            }
        
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            return {
                'error': 'invalid_image',
                'message': 'Failed to load image'
            }
        
        return self.predict_from_array(img)
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict gesture from image bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                'error': 'invalid_image',
                'message': 'Failed to decode image'
            }
        
        return self.predict_from_array(img)


# Global instance (lazy loaded)
_gesture_classifier: Optional[GestureClassifier] = None


def get_gesture_classifier() -> GestureClassifier:
    """Get or create global gesture classifier instance."""
    global _gesture_classifier
    if _gesture_classifier is None:
        _gesture_classifier = GestureClassifier()
    return _gesture_classifier


def predict_gesture(image_path: str) -> Dict[str, Any]:
    """
    Convenience function to predict gesture from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Prediction results
    """
    classifier = get_gesture_classifier()
    return classifier.predict_from_file(image_path)
