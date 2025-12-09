"""
Reorganize Kaggle gesture dataset properly.
The dataset has images in individual sequence folders, not by class.
"""

import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

# Paths
dataset_root = Path(__file__).parent.parent / "datasets" / "dataset_hand_gesture"
train_csv = dataset_root / "train.csv"
val_csv = dataset_root / "val.csv"

# Output directories
new_train_dir = dataset_root / "train_organized"
new_val_dir = dataset_root / "val_organized"

# Backup
backup_dir = dataset_root / "backup_originals"
backup_dir.mkdir(exist_ok=True)

print("=" * 70)
print("REORGANIZING GESTURE DATASET")
print("=" * 70)

# Class mapping
CLASS_MAPPING = {
    'Left_Swipe_new': 'left_swipe',
    'Left Swipe_new': 'left_swipe',
    'Left Swipe_new_Left Swipe_new': 'left_swipe',
    'Right_Swipe_new': 'right_swipe',
    'Right Swipe_new': 'right_swipe',
    'Stop_new': 'stop',
    'Stop Gesture_new': 'stop',
    'Thumbs_Up_new': 'thumbs_up',
    'Thumbs Up_new': 'thumbs_up',
    'Thumbs_Down_new': 'thumbs_down',
    'Thumbs Down_new': 'thumbs_down',
}

def extract_class_from_folder(folder_name):
    """Extract gesture class from folder name."""
    for old_class, new_class in CLASS_MAPPING.items():
        if old_class in folder_name:
            return new_class
    return None

def find_all_images():
    """Find all images in the dataset."""
    all_images = []
    
    # Search in train directory
    train_base = dataset_root / "train" / "train"
    if train_base.exists():
        for folder in train_base.iterdir():
            if folder.is_dir():
                gesture_class = extract_class_from_folder(folder.name)
                if gesture_class:
                    for img_file in folder.glob("*.jpg"):
                        all_images.append({
                            'path': img_file,
                            'class': gesture_class,
                            'split': 'train'
                        })
    
    # Search in val directory
    val_base = dataset_root / "val" / "val"
    if val_base.exists():
        for folder in val_base.iterdir():
            if folder.is_dir():
                gesture_class = extract_class_from_folder(folder.name)
                if gesture_class:
                    for img_file in folder.glob("*.jpg"):
                        all_images.append({
                            'path': img_file,
                            'class': gesture_class,
                            'split': 'val'
                        })
    
    return all_images

print("\n1. Scanning dataset...")
all_images = find_all_images()
print(f"   Found {len(all_images)} images")

# Count by class
class_counts = Counter(img['class'] for img in all_images)
print("\n2. Class distribution:")
for cls, count in sorted(class_counts.items()):
    print(f"   {cls:15s}: {count:4d} images")

# Remove duplicates based on filename
unique_images = {}
for img in all_images:
    key = img['path'].name
    if key not in unique_images:
        unique_images[key] = img
    
all_images = list(unique_images.values())
print(f"\n3. After removing duplicates: {len(all_images)} images")

# Shuffle and split 80/20
random.seed(42)
images_by_class = defaultdict(list)
for img in all_images:
    images_by_class[img['class']].append(img)

# Balance classes to minimum size
min_size = min(len(imgs) for imgs in images_by_class.values())
print(f"\n4. Minimum class size: {min_size}")
print(f"   Balancing all classes to {min_size} images each")

balanced_images = []
for cls, imgs in images_by_class.items():
    # Shuffle and take minimum size
    random.shuffle(imgs)
    balanced_images.extend(imgs[:min_size])

random.shuffle(balanced_images)

# Split 80/20
split_idx = int(len(balanced_images) * 0.8)
train_images = balanced_images[:split_idx]
val_images = balanced_images[split_idx:]

print(f"\n5. Final split:")
print(f"   Train: {len(train_images)} images")
print(f"   Val: {len(val_images)} images")

# Create organized directories
print("\n6. Creating organized structure...")
for cls in CLASS_MAPPING.values():
    (new_train_dir / cls).mkdir(parents=True, exist_ok=True)
    (new_val_dir / cls).mkdir(parents=True, exist_ok=True)

# Copy images to organized structure
print("\n7. Copying images to organized folders...")

train_count = 0
for img in train_images:
    dst = new_train_dir / img['class'] / img['path'].name
    if not dst.exists():
        shutil.copy2(img['path'], dst)
        train_count += 1

val_count = 0
for img in val_images:
    dst = new_val_dir / img['class'] / img['path'].name
    if not dst.exists():
        shutil.copy2(img['path'], dst)
        val_count += 1

print(f"   Copied {train_count} train images")
print(f"   Copied {val_count} val images")

# Create new CSV files
class_to_label = {
    'left_swipe': 0,
    'right_swipe': 1,
    'stop': 2,
    'thumbs_down': 3,
    'thumbs_up': 4
}

print("\n8. Creating new CSV files...")
new_train_csv = dataset_root / "train_final.csv"
new_val_csv = dataset_root / "val_final.csv"

with open(new_train_csv, 'w', encoding='utf-8') as f:
    for img in train_images:
        img_name = img['path'].stem  # without extension
        cls = img['class']
        label = class_to_label[cls]
        f.write(f"{img_name};{cls};{label}\n")

with open(new_val_csv, 'w', encoding='utf-8') as f:
    for img in val_images:
        img_name = img['path'].stem
        cls = img['class']
        label = class_to_label[cls]
        f.write(f"{img_name};{cls};{label}\n")

print(f"   ✅ Created: {new_train_csv}")
print(f"   ✅ Created: {new_val_csv}")

# Show final distribution
train_class_counts = Counter(img['class'] for img in train_images)
val_class_counts = Counter(img['class'] for img in val_images)

print("\n9. Final class distribution:")
print("\nTrain:")
for cls in sorted(train_class_counts.keys()):
    print(f"   {cls:15s}: {train_class_counts[cls]:4d} images")

print("\nVal:")
for cls in sorted(val_class_counts.keys()):
    print(f"   {cls:15s}: {val_class_counts[cls]:4d} images")

print("\n" + "=" * 70)
print("✅ DATASET REORGANIZATION COMPLETE!")
print("=" * 70)
print(f"\nNew directories:")
print(f"  Train: {new_train_dir}")
print(f"  Val: {new_val_dir}")
print(f"\nNew CSV files:")
print(f"  Train: {new_train_csv}")
print(f"  Val: {new_val_csv}")
print(f"\n💡 Now update train_gesture_model.py to use these new files")
