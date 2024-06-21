import cv2
import mediapipe as mp
import pickle
import numpy as np
from gtts import gTTS
import pygame
import os
import threading
import time

# Load the trained model
with open("hand_gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio playback
pygame.mixer.init()

# Function to speak the recognized gesture asynchronously
def speak_text(text):
    def speak():
        tts = gTTS(text=text, lang='en')
        filename = "temp.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        os.remove(filename)
    
    threading.Thread(target=speak).start()

# Initialize webcam
cap = cv2.VideoCapture(0)

previous_gesture = ""
cooldown = 2  # Cooldown period in seconds
last_spoken_time = time.time() - cooldown

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and detect hands
    results = hands.process(image)

    # Draw hand landmarks on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).flatten().reshape(1, -1)

            # Predict gesture
            gesture = model.predict(landmarks)[0]

            # Get the dimensions of the image
            height, width, _ = image.shape

            # Define the position for the subtitle (center bottom)
            text_size = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50

            # Draw the subtitle
            cv2.putText(image, gesture, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

            # Speak the recognized gesture if it's different from the previous one and cooldown period has passed
            current_time = time.time()
            if gesture != previous_gesture and (current_time - last_spoken_time) >= cooldown:
                speak_text(gesture)
                previous_gesture = gesture
                last_spoken_time = current_time

    # Display the image
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
