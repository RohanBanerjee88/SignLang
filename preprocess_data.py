import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

data = []
labels = []

gestures = ["Hello", "Love", "Food", "Rain"]  # List your gestures here

for gesture in gestures:
    folder = f"../test_v2/gestures/{gesture}"
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        continue
    
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        print(f"Processing file: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"Error converting image {image_path}: {e}")
            continue

        # Process the image and detect hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                data.append(landmarks)
                labels.append(gesture)
        else:
            print(f"No hand landmarks found in image {image_path}")

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Save the data and labels for later use
with open("hand_gesture_data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

print(f"Data and labels saved. Total samples: {len(data)}")
