"""
Script to fix and normalize gesture dataset.
- Normalizes class names
- Removes duplicates
- Balances classes
- Creates proper train/val split
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from collections import Counter

# Paths
dataset_root = Path(__file__).parent.parent / "datasets" / "dataset_hand_gesture"
train_dir = dataset_root / "train"
val_dir = dataset_root / "val"
train_csv = dataset_root / "train.csv"
val_csv = dataset_root / "val.csv"

# Backup original files
backup_dir = dataset_root / "backup_original"
backup_dir.mkdir(exist_ok=True)

if train_csv.exists():
    shutil.copy(train_csv, backup_dir / "train_original.csv")
if val_csv.exists():
    shutil.copy(val_csv, backup_dir / "val_original.csv")

print("=" * 60)
print("FIXING GESTURE DATASET")
print("=" * 60)

# Class name mapping to normalize
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

def load_and_fix_csv(csv_path, img_dir):
    """Load CSV and fix class names."""
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 3:
                img_name = parts[0]
                old_class = parts[1]
                label = parts[2]
                
                # Normalize class name
                new_class = CLASS_MAPPING.get(old_class, old_class.lower().replace(' ', '_'))
                
                # Verify image exists
                img_path = img_dir / old_class / f"{img_name}.jpg"
                if not img_path.exists():
                    # Try with normalized class name
                    for possible_class in CLASS_MAPPING.keys():
                        alt_path = img_dir / possible_class / f"{img_name}.jpg"
                        if alt_path.exists():
                            img_path = alt_path
                            break
                
                if img_path.exists():
                    data.append({
                        'image': img_name,
                        'class': new_class,
                        'label': label,
                        'path': str(img_path.relative_to(dataset_root))
                    })
    
    return pd.DataFrame(data)

# Load data
print("\n1. Loading and normalizing data...")
train_df = load_and_fix_csv(train_csv, train_dir)
val_df = load_and_fix_csv(val_csv, val_dir)

print(f"   - Train: {len(train_df)} images")
print(f"   - Val: {len(val_df)} images")

# Check class distribution
print("\n2. Original class distribution:")
print("\nTrain:")
print(train_df['class'].value_counts())
print("\nVal:")
print(val_df['class'].value_counts())

# Combine all data
all_data = pd.concat([train_df, val_df], ignore_index=True)
print(f"\n3. Total images: {len(all_data)}")

# Remove duplicates based on image name
print("\n4. Removing duplicates...")
all_data = all_data.drop_duplicates(subset=['image'], keep='first')
print(f"   After deduplication: {len(all_data)} images")

# Check for class balance
class_counts = all_data['class'].value_counts()
print("\n5. Class distribution after cleanup:")
print(class_counts)

# Find minimum class size
min_class_size = class_counts.min()
print(f"\n6. Minimum class size: {min_class_size}")

# Balance classes by undersampling
print("\n7. Balancing classes...")
balanced_data = []
for class_name in class_counts.index:
    class_data = all_data[all_data['class'] == class_name]
    # Sample equally from each class
    sampled = class_data.sample(n=min(len(class_data), min_class_size * 3), random_state=42)
    balanced_data.append(sampled)

balanced_df = pd.concat(balanced_data, ignore_index=True)
print(f"   Balanced dataset: {len(balanced_df)} images")
print("\nBalanced class distribution:")
print(balanced_df['class'].value_counts())

# Create proper train/val split (80/20)
print("\n8. Creating 80/20 train/val split...")
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(
    balanced_df, 
    test_size=0.2, 
    stratify=balanced_df['class'],
    random_state=42
)

print(f"   - Train: {len(train_data)} images")
print(f"   - Val: {len(val_data)} images")

# Remap labels to 0-4
class_to_label = {
    'left_swipe': 0,
    'right_swipe': 1,
    'stop': 2,
    'thumbs_down': 3,
    'thumbs_up': 4
}

train_data['label'] = train_data['class'].map(class_to_label)
val_data['label'] = val_data['class'].map(class_to_label)

# Save new CSVs
print("\n9. Saving cleaned datasets...")
train_output = dataset_root / "train_clean.csv"
val_output = dataset_root / "val_clean.csv"

with open(train_output, 'w', encoding='utf-8') as f:
    for _, row in train_data.iterrows():
        f.write(f"{row['image']};{row['class']};{row['label']}\n")

with open(val_output, 'w', encoding='utf-8') as f:
    for _, row in val_data.iterrows():
        f.write(f"{row['image']};{row['class']};{row['label']}\n")

print(f"   ✅ Saved: {train_output}")
print(f"   ✅ Saved: {val_output}")

# Rename old files and use new ones
print("\n10. Updating dataset files...")
if train_csv.exists():
    train_csv.rename(dataset_root / "train_old.csv")
if val_csv.exists():
    val_csv.rename(dataset_root / "val_old.csv")

train_output.rename(train_csv)
val_output.rename(val_csv)

print("\n" + "=" * 60)
print("✅ DATASET FIXED SUCCESSFULLY!")
print("=" * 60)
print(f"\nFinal train set: {len(train_data)} images")
print(f"Final val set: {len(val_data)} images")
print("\nClass distribution (train):")
print(train_data['class'].value_counts())
print("\nClass distribution (val):")
print(val_data['class'].value_counts())
print("\n💡 Original files backed up to:", backup_dir)
