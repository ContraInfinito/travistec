"""
Train Gesture Recognition with LeapGestRecog Dataset
High-quality dataset: 20,000 images, 10 gestures, 10 people
Expected accuracy: 90%+
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
import keras
from keras import layers, models, optimizers, callbacks
from keras.applications import MobileNetV3Large
import cv2
from sklearn.model_selection import train_test_split
import random

# Configuration
IMG_SIZE = (160, 160)  # Optimal for MobileNet
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.15

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "leapGestRecog"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Map LeapGest gestures to our needed gestures
GESTURE_MAPPING = {
    '01_palm': 'stop',           # Open palm = stop
    '05_thumb': 'thumbs_up',     # Thumb up
    '10_down': 'thumbs_down',    # Thumb down
    '02_l': 'left_swipe',        # L gesture = left
    '06_index': 'right_swipe',   # Index = right (pointing)
}

# Our 5 classes
CLASSES = ['left_swipe', 'right_swipe', 'stop', 'thumbs_down', 'thumbs_up']

print("=" * 70)
print("TRAINING GESTURE RECOGNITION - LeapGestRecog Dataset")
print("=" * 70)
print(f"\nDataset: {DATASET_DIR}")
print(f"Target gestures: {len(CLASSES)}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}\n")

class GestureDataGenerator(keras.utils.Sequence):
    """Data generator for LeapGestRecog dataset."""
    
    def __init__(self, image_paths, labels, batch_size=32, img_size=(160, 160), 
                 augment=True, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        X, y = self._generate_batch(batch_paths, batch_labels)
        return X, y
    
    def _load_and_preprocess(self, img_path):
        """Load and preprocess image."""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((*self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LANCZOS4)
        
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random rotation (-15 to +15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Random brightness (0.8 to 1.2)
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # Random contrast (0.8 to 1.2)
            if random.random() > 0.5:
                contrast = random.uniform(0.8, 1.2)
                mean = img.mean()
                img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Apply CLAHE for better contrast
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Normalize to [-1, 1] for MobileNetV3
        img = (img.astype(np.float32) / 127.5) - 1.0
        
        return img
    
    def _generate_batch(self, batch_paths, batch_labels):
        X = np.zeros((len(batch_paths), *self.img_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_paths), len(CLASSES)), dtype=np.float32)
        
        for i, (img_path, label) in enumerate(zip(batch_paths, batch_labels)):
            X[i] = self._load_and_preprocess(img_path)
            y[i, label] = 1.0
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_model(input_shape, num_classes):
    """Create MobileNetV3Large model."""
    
    # Base model
    base_model = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=False
    )
    
    # Freeze early layers, unfreeze last 40 for fine-tuning
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    for layer in base_model.layers[-40:]:
        layer.trainable = True
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    return model

# Load dataset
print("1. Loading LeapGestRecog dataset...")
all_images = []
all_labels = []

# Scan directories
for person_dir in sorted(DATASET_DIR.glob("[0-9][0-9]")):
    person_id = person_dir.name
    
    for gesture_dir in person_dir.iterdir():
        if not gesture_dir.is_dir():
            continue
        
        gesture_name = gesture_dir.name
        
        # Check if this gesture is in our mapping
        if gesture_name not in GESTURE_MAPPING:
            continue
        
        our_gesture = GESTURE_MAPPING[gesture_name]
        label = CLASSES.index(our_gesture)
        
        # Get all images
        for img_file in gesture_dir.glob("*.png"):
            all_images.append(str(img_file))
            all_labels.append(label)

print(f"   Total images loaded: {len(all_images)}")
print(f"\n   Distribution:")
for i, cls in enumerate(CLASSES):
    count = all_labels.count(i)
    print(f"   {cls:15s}: {count:5d} images")

# Split train/val
print(f"\n2. Creating train/validation split ({int((1-VALIDATION_SPLIT)*100)}/{int(VALIDATION_SPLIT*100)})...")
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels,
    test_size=VALIDATION_SPLIT,
    stratify=all_labels,
    random_state=42
)

print(f"   Train: {len(train_images)} images")
print(f"   Val: {len(val_images)} images")

# Create generators
print("\n3. Creating data generators...")
train_gen = GestureDataGenerator(
    train_images, train_labels,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=True,
    shuffle=True
)

val_gen = GestureDataGenerator(
    val_images, val_labels,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=False,
    shuffle=False
)

print(f"   Train batches: {len(train_gen)}")
print(f"   Val batches: {len(val_gen)}")

# Create model
print("\n4. Creating MobileNetV3Large model...")
model = create_model((*IMG_SIZE, 3), len(CLASSES))

# Compile
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print(f"   Total parameters: {model.count_params():,}")
trainable_count = sum(w.shape.num_elements() for w in model.trainable_weights)
print(f"   Trainable parameters: {trainable_count:,}")

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / "gesture_leap_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n5. Training model...")
print(f"   Starting training for {EPOCHS} epochs...\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_list,
    verbose=1
)

# Save final model
final_model_path = MODELS_DIR / "gesture_leap_final.keras"
model.save(final_model_path)
print(f"\n✅ Final model saved: {final_model_path}")

# Save classes
classes_path = MODELS_DIR / "gesture_classes.txt"
with open(classes_path, 'w') as f:
    for cls in CLASSES:
        f.write(f"{cls}\n")
print(f"✅ Classes saved: {classes_path}")

# Final evaluation
print("\n6. Final Evaluation:")
val_loss, val_acc, val_top2 = model.evaluate(val_gen, verbose=0)
print(f"   ✅ Validation Loss: {val_loss:.4f}")
print(f"   ✅ Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   ✅ Top-2 Accuracy: {val_top2:.4f} ({val_top2*100:.2f}%)")

# Training summary
best_epoch = np.argmax(history.history['val_accuracy']) + 1
best_acc = max(history.history['val_accuracy'])
print(f"\n   Best epoch: {best_epoch}")
print(f"   Best val accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Results Summary:")
print(f"   Dataset: LeapGestRecog (20,000 images)")
print(f"   Model: MobileNetV3Large")
print(f"   Final Accuracy: {val_acc*100:.2f}%")
print(f"   Best Accuracy: {best_acc*100:.2f}%")
print(f"\n💾 Models saved:")
print(f"   - Best: {MODELS_DIR / 'gesture_leap_best.keras'}")
print(f"   - Final: {final_model_path}")
print(f"\n🎯 Next step: Update gesture_classifier.py to use the new model!")
