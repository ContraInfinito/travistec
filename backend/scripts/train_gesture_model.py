"""
Train hand gesture recognition models using Transfer Learning.

This script trains two models:
1. ResNet50 - Deep residual network
2. MobileNetV2 - Lightweight efficient network

Dataset: dataset_hand_gesture with 5 classes:
- Left_Swipe
- Right_Swipe
- Stop
- Thumbs_Down
- Thumbs_Up

The models are saved to backend/models/ as:
- gesture_resnet50.keras
- gesture_mobilenetv2.keras
- gesture_classes.txt (class labels)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Configuration
DATASET_DIR = backend_dir / "datasets" / "dataset_hand_gesture"
MODELS_DIR = backend_dir / "models"
TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
IMG_SIZE = (160, 160)  # Optimizado: reducido de 224x224 para procesamiento más rápido
BATCH_SIZE = 64  # Optimizado: aumentado de 32 para procesar más imágenes por iteración
EPOCHS = 12  # Optimizado: reducido de 15 (Early Stopping detendrá antes si es necesario)
LEARNING_RATE = 0.0001

# Class mapping - normalize class names
CLASS_MAPPING = {
    'Left_Swipe_new': 'left_swipe',
    'Left Swipe_new_Left Swipe_new': 'left_swipe',
    'Right_Swipe_new': 'right_swipe',
    'Right Swipe_new': 'right_swipe',
    'Stop_new': 'stop',
    'Stop Gesture_new': 'stop',
    'Thumbs_Down_new': 'thumbs_down',
    'Thumbs Down_new': 'thumbs_down',
    'Thumbs_Up_new': 'thumbs_up',
    'Thumbs Up_new': 'thumbs_up',
}


def load_dataset_from_csv(csv_path, base_dir):
    """Load dataset from CSV file with folder;class;id format."""
    print(f"Loading dataset from {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Read CSV (format: folder_name;class_name;class_id)
    df = pd.read_csv(csv_path, sep=';', header=None, names=['folder', 'class', 'id'])
    
    # Normalize class names
    df['class_normalized'] = df['class'].map(CLASS_MAPPING)
    
    # Remove any rows with unmapped classes
    df = df.dropna(subset=['class_normalized'])
    
    # Build image paths
    data = []
    train_dir = base_dir / "train" / "train"
    val_dir = base_dir / "val" / "val"
    
    # Try both train and val directories
    for _, row in df.iterrows():
        folder_name = row['folder']
        class_name = row['class_normalized']
        
        # Try train directory first
        folder_path = train_dir / folder_name
        if not folder_path.exists():
            # Try val directory
            folder_path = val_dir / folder_name
        
        if folder_path.exists():
            # Get all images in folder
            for img_path in folder_path.glob("*.png"):
                data.append({
                    'path': str(img_path),
                    'class': class_name
                })
    
    result_df = pd.DataFrame(data)
    print(f"Loaded {len(result_df)} images with {result_df['class'].nunique()} classes")
    print(f"Class distribution:\n{result_df['class'].value_counts()}")
    
    return result_df


def create_data_generators(train_df, val_df):
    """Create augmented data generators for training and validation."""
    
    # Training data augmentation (optimizado: reducido para procesamiento más rápido)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,  # Reducido de 20
        width_shift_range=0.15,  # Reducido de 0.2
        height_shift_range=0.15,  # Reducido de 0.2
        horizontal_flip=True,
        zoom_range=0.15,  # Reducido de 0.2
        fill_mode='nearest'
        # Removido shear_range para acelerar
    )
    
    # Validation data (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='path',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator


def build_resnet50_model(num_classes):
    """Build ResNet50-based model for gesture classification."""
    print("\n=== Building ResNet50 Model ===")
    
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    return model


def build_mobilenetv2_model(num_classes):
    """Build MobileNetV2-based model for gesture classification."""
    print("\n=== Building MobileNetV2 Model ===")
    
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    return model


def train_model(model, train_gen, val_gen, model_name):
    """Train the model with callbacks."""
    print(f"\n=== Training {model_name} ===")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history


def evaluate_model(model, val_gen, model_name):
    """Evaluate model on validation set."""
    print(f"\n=== Evaluating {model_name} ===")
    results = model.evaluate(val_gen, verbose=1)
    
    print(f"\n{model_name} Results:")
    print(f"  Validation Loss: {results[0]:.4f}")
    print(f"  Validation Accuracy: {results[1]:.4f}")
    print(f"  Top-2 Accuracy: {results[2]:.4f}")
    
    return results


def save_model_and_metadata(model, class_indices, model_name, metrics):
    """Save trained model and metadata."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Save model
    model_path = MODELS_DIR / f"{model_name}.keras"
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save class labels (only once)
    classes_path = MODELS_DIR / "gesture_classes.txt"
    if not classes_path.exists():
        # Sort by index
        sorted_classes = sorted(class_indices.items(), key=lambda x: x[1])
        with open(classes_path, 'w') as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")
        print(f"✅ Class labels saved to: {classes_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'num_classes': len(class_indices),
        'classes': {name: idx for name, idx in class_indices.items()},
        'input_shape': [*IMG_SIZE, 3],
        'metrics': {
            'val_loss': float(metrics[0]),
            'val_accuracy': float(metrics[1]),
            'top_2_accuracy': float(metrics[2])
        }
    }
    
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("GESTURE RECOGNITION MODEL TRAINING")
    print("=" * 80)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} device(s)")
    else:
        print("⚠️  Training on CPU (this will be slower)")
    
    # Load datasets
    print("\n--- Loading Datasets ---")
    train_df = load_dataset_from_csv(TRAIN_CSV, DATASET_DIR)
    val_df = load_dataset_from_csv(VAL_CSV, DATASET_DIR)
    
    # Create data generators
    print("\n--- Creating Data Generators ---")
    train_gen, val_gen = create_data_generators(train_df, val_df)
    num_classes = len(train_gen.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_gen.class_indices}")
    
    # Store class indices for later
    class_indices = train_gen.class_indices
    
    # === Train ResNet50 ===
    print("\n" + "=" * 80)
    print("TRAINING MODEL 1: ResNet50")
    print("=" * 80)
    resnet_model = build_resnet50_model(num_classes)
    resnet_model.summary()
    
    train_model(resnet_model, train_gen, val_gen, "gesture_resnet50")
    resnet_metrics = evaluate_model(resnet_model, val_gen, "gesture_resnet50")
    save_model_and_metadata(resnet_model, class_indices, "gesture_resnet50", resnet_metrics)
    
    # === Train MobileNetV2 ===
    print("\n" + "=" * 80)
    print("TRAINING MODEL 2: MobileNetV2")
    print("=" * 80)
    mobilenet_model = build_mobilenetv2_model(num_classes)
    mobilenet_model.summary()
    
    train_model(mobilenet_model, train_gen, val_gen, "gesture_mobilenetv2")
    mobilenet_metrics = evaluate_model(mobilenet_model, val_gen, "gesture_mobilenetv2")
    save_model_and_metadata(mobilenet_model, class_indices, "gesture_mobilenetv2", mobilenet_metrics)
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print("\n📊 Final Results:")
    print(f"\nResNet50:")
    print(f"  Accuracy: {resnet_metrics[1]:.2%}")
    print(f"  Top-2 Accuracy: {resnet_metrics[2]:.2%}")
    print(f"\nMobileNetV2:")
    print(f"  Accuracy: {mobilenet_metrics[1]:.2%}")
    print(f"  Top-2 Accuracy: {mobilenet_metrics[2]:.2%}")
    
    print(f"\n✅ Models saved to: {MODELS_DIR}")
    print("\n🎯 Next steps:")
    print("  1. Backend will automatically load these models on startup")
    print("  2. Use endpoint: POST /api/v1/classify/gesture")
    print("  3. Frontend will display gesture predictions alongside emotions")


if __name__ == "__main__":
    main()
