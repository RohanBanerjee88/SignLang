import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

gesture_name = "Rain"  # Change this for each gesture
gesture_folder = f"../test_v2/gestures/{gesture_name}"
os.makedirs(gesture_folder, exist_ok=True)

image_count = 0

print("Press 'space' to capture an image. Press 'ESC' to exit.")

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

    # Display the image
    cv2.imshow('Hand Detection', image)

    key = cv2.waitKey(5)
    if key & 0xFF == ord(' '):  # Spacebar to capture the image
        if results.multi_hand_landmarks:
            image_count += 1
            cv2.imwrite(f"{gesture_folder}/{gesture_name}_{image_count}.jpg", image)
            print(f"Captured image {image_count}")
        else:
            print("No hand detected, try again.")
    elif key & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
