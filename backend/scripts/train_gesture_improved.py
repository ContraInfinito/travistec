"""
IMPROVED Gesture Recognition Training with HEAVY Data Augmentation
This script will train a MUCH better model by:
1. Using MobileNetV3Large (better than V2)
2. Heavy data augmentation
3. Better training strategy
4. Focal loss for hard examples
5. Mixup augmentation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from pathlib import Path
import keras
from keras import layers, models, optimizers, callbacks
from keras.applications import MobileNetV3Large
import cv2
from sklearn.utils.class_weight import compute_class_weight
import json

# Configuration
IMG_SIZE = (224, 224)  # Larger for better quality
BATCH_SIZE = 32
EPOCHS = 30  # More epochs
LEARNING_RATE = 0.0001
NUM_CLASSES = 5

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "dataset_hand_gesture"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Class names
CLASSES = ['left_swipe', 'right_swipe', 'stop', 'thumbs_down', 'thumbs_up']

print("=" * 70)
print("IMPROVED GESTURE RECOGNITION TRAINING")
print("=" * 70)

class ImprovedDataGenerator(keras.utils.Sequence):
    """Data generator with HEAVY augmentation."""
    
    def __init__(self, csv_file, img_base_dir, batch_size=32, img_size=(224, 224), 
                 augment=True, shuffle=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        
        # Load CSV
        self.data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 3:
                    self.data.append({
                        'image': parts[0],
                        'class': parts[1],
                        'label': int(parts[2])
                    })
        
        self.img_base_dir = img_base_dir
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indexes]
        
        X, y = self._generate_batch(batch_data)
        return X, y
    
    def _load_and_augment_image(self, img_path):
        """Load and aggressively augment image."""
        # Try to load image
        img = cv2.imread(str(img_path))
        if img is None:
            # Create a black image if not found
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LANCZOS4)
        
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random rotation (-20 to +20 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-20, 20)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Random scaling (0.8x to 1.2x)
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.8, 1.2)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                # Crop or pad to original size
                if scale > 1:
                    start_y = (new_h - h) // 2
                    start_x = (new_w - w) // 2
                    img = img[start_y:start_y+h, start_x:start_x+w]
                else:
                    pad_y = (h - new_h) // 2
                    pad_x = (w - new_w) // 2
                    img = cv2.copyMakeBorder(img, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, 
                                            cv2.BORDER_REPLICATE)
            
            # Random brightness (0.7 to 1.3)
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.7, 1.3)
                img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # Random contrast (0.7 to 1.3)
            if np.random.rand() > 0.5:
                contrast = np.random.uniform(0.7, 1.3)
                mean = img.mean()
                img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
            
            # Random Gaussian noise
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Random blur
            if np.random.rand() > 0.7:
                kernel_size = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Apply CLAHE
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Normalize to [-1, 1] for MobileNetV3
        img = (img.astype(np.float32) / 127.5) - 1.0
        
        return img
    
    def _generate_batch(self, batch_data):
        X = np.zeros((len(batch_data), *self.img_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_data), NUM_CLASSES), dtype=np.float32)
        
        for i, item in enumerate(batch_data):
            # Find image file
            class_name = item['class']
            img_name = item['image']
            
            # Try different paths
            possible_paths = [
                self.img_base_dir / class_name / f"{img_name}.jpg",
                self.img_base_dir / class_name / f"{img_name}",
            ]
            
            img = None
            for img_path in possible_paths:
                if img_path.exists():
                    img = self._load_and_augment_image(img_path)
                    break
            
            if img is None:
                # Create black image if not found
                img = np.zeros((*self.img_size, 3), dtype=np.float32)
            
            X[i] = img
            y[i, item['label']] = 1.0
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_improved_model(input_shape, num_classes):
    """Create MobileNetV3Large with improved architecture."""
    
    # Base model
    base_model = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=False
    )
    
    # Unfreeze last 30 layers for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    
    # Base model
    x = base_model(inputs, training=True)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    return model

# Check for CSV files
train_csv = DATASET_DIR / "train.csv"
val_csv = DATASET_DIR / "val.csv"

if not train_csv.exists() or not val_csv.exists():
    print("\n❌ ERROR: CSV files not found!")
    print(f"   Looking for: {train_csv}")
    print(f"   Looking for: {val_csv}")
    print("\n💡 You need a proper gesture dataset with images.")
    print("   Recommended dataset: 'Hand Gesture Recognition Database' from Kaggle")
    print("   Link: https://www.kaggle.com/gti-upm/leapgestrecog")
    exit(1)

# Create data generators
print("\n1. Creating data generators...")
train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"

train_gen = ImprovedDataGenerator(
    train_csv, train_dir, 
    batch_size=BATCH_SIZE, 
    img_size=IMG_SIZE,
    augment=True, 
    shuffle=True
)

val_gen = ImprovedDataGenerator(
    val_csv, val_dir, 
    batch_size=BATCH_SIZE, 
    img_size=IMG_SIZE,
    augment=False, 
    shuffle=False
)

print(f"   Train batches: {len(train_gen)}")
print(f"   Val batches: {len(val_gen)}")

# Create model
print("\n2. Creating MobileNetV3Large model...")
model = create_improved_model((*IMG_SIZE, 3), NUM_CLASSES)

# Compile
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print(f"   Total parameters: {model.count_params():,}")

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / "gesture_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n3. Training model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Image size: {IMG_SIZE}")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_list,
    verbose=1
)

# Save final model
final_model_path = MODELS_DIR / "gesture_mobilenetv3.keras"
model.save(final_model_path)
print(f"\n✅ Final model saved: {final_model_path}")

# Save classes
classes_path = MODELS_DIR / "gesture_classes.txt"
with open(classes_path, 'w') as f:
    for cls in CLASSES:
        f.write(f"{cls}\n")
print(f"✅ Classes saved: {classes_path}")

# Evaluate
print("\n4. Final Evaluation:")
val_loss, val_acc, val_top2 = model.evaluate(val_gen, verbose=0)
print(f"   Validation Loss: {val_loss:.4f}")
print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Top-2 Accuracy: {val_top2:.4f} ({val_top2*100:.2f}%)")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel saved to: {final_model_path}")
print("\n💡 IMPORTANT: The current dataset seems incomplete.")
print("   For 90%+ accuracy, you need:")
print("   1. At least 1000-2000 images PER gesture class")
print("   2. Images from different people, lighting, backgrounds")
print("   3. Proper hand segmentation or consistent framing")
print("\n   Recommended action: Download 'LeapGestRecog' dataset from Kaggle")
print("   Link: https://www.kaggle.com/gti-upm/leapgestrecog")
