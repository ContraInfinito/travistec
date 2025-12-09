"""
Dataset Creator - Capture your own gesture images with MediaPipe
Run this to create a professional dataset with proper hand isolation
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
GESTURES = ['left_swipe', 'right_swipe', 'stop', 'thumbs_down', 'thumbs_up']
IMAGES_PER_GESTURE = 500  # Minimum for good accuracy
OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "my_gestures"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def capture_gesture_dataset():
    """Interactive gesture capture tool."""
    
    # Create directories
    for gesture in GESTURES:
        (OUTPUT_DIR / gesture).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_gesture_idx = 0
    current_gesture = GESTURES[current_gesture_idx]
    count = 0
    capturing = False
    
    print("=" * 70)
    print("GESTURE DATASET CREATOR")
    print("=" * 70)
    print("\nControls:")
    print("  SPACE - Start/Stop capturing")
    print("  N - Next gesture")
    print("  Q - Quit")
    print(f"\nCapture {IMAGES_PER_GESTURE} images per gesture")
    print("=" * 70)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract hand region
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)) - 50, int(max(x_coords)) + 50
                y_min, y_max = int(min(y_coords)) - 50, int(max(y_coords)) + 50
                
                # Clamp to frame bounds
                x_min, x_max = max(0, x_min), min(w, x_max)
                y_min, y_max = max(0, y_min), min(h, y_max)
                
                # Draw bounding box
                color = (0, 255, 0) if capturing else (255, 0, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Capture image
                if capturing:
                    hand_roi = rgb_frame[y_min:y_max, x_min:x_max]
                    
                    if hand_roi.size > 0:
                        # Resize to 224x224
                        hand_roi_resized = cv2.resize(hand_roi, (224, 224))
                        
                        # Save
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = OUTPUT_DIR / current_gesture / f"{timestamp}.jpg"
                        cv2.imwrite(str(filename), cv2.cvtColor(hand_roi_resized, cv2.COLOR_RGB2BGR))
                        
                        count += 1
                        
                        if count >= IMAGES_PER_GESTURE:
                            capturing = False
                            print(f"\n✅ Completed {current_gesture}: {count} images")
                            count = 0
                            
                            # Move to next gesture
                            current_gesture_idx += 1
                            if current_gesture_idx < len(GESTURES):
                                current_gesture = GESTURES[current_gesture_idx]
                                print(f"\n👉 Next gesture: {current_gesture}")
                            else:
                                print("\n🎉 ALL GESTURES COMPLETE!")
                                break
        
        # Display info
        status = "CAPTURING" if capturing else "PAUSED"
        cv2.putText(frame, f"Gesture: {current_gesture}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {status}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count}/{IMAGES_PER_GESTURE}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=Start/Stop | N=Next | Q=Quit", (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Gesture Dataset Creator', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            capturing = not capturing
            if capturing:
                print(f"\n🔴 Capturing {current_gesture}...")
        elif key == ord('n'):
            if not capturing:
                current_gesture_idx = (current_gesture_idx + 1) % len(GESTURES)
                current_gesture = GESTURES[current_gesture_idx]
                count = 0
                print(f"\n👉 Switched to: {current_gesture}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE!")
    print("=" * 70)
    print(f"\nImages saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review images and delete bad ones")
    print("2. Run train_gesture_improved.py with this dataset")

if __name__ == "__main__":
    try:
        capture_gesture_dataset()
    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
