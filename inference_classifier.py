import pickle
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open a video capture device (change the index if needed)
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

heatmap = None  # Initialize an empty heatmap

labels_dict = {0: 'FanOff', 1: 'FanOn', 2: 'LightOn', 3: 'LightOff', 4: 'TVOn', 5: 'TVOff'}
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

engine = pyttsx3.init()


while True:
    data_aux = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for idx,hand_landmarks in enumerate(results.multi_hand_landmarks):
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * W)
                y = int(landmark.y * H)
                landmarks.append([x, y])

                # for i, landmark in enumerate(landmarks[:-1]):
                #     cv2.circle(frame, tuple(landmark), 5, (255, 0, 0), -1)  # Plot a point for each landmark
                #     cv2.line(frame, tuple(landmark), tuple(landmarks[i + 1]), (255, 0, 0), 2)  # Connect landmarks with lines
                
                # cv2.line(frame, tuple(landmarks[-1]), tuple(landmarks[0]), (255, 0, 0), 2)

                cv2.circle(frame, tuple([x, y]), 5, (0, 255, 0), -1)  # Plot a point for each landmark

                for i in range(len(landmarks)-1):
                    cv2.line(frame, tuple(landmarks[i]), tuple(landmarks[i+1]), colors[idx], 2)

                cv2.line(frame, tuple(landmarks[-1]), tuple(landmarks[0]),colors[idx], 2)


        # Create an empty frame heatmap
        frame_heatmap = np.zeros((H, W))

        # Add landmarks to the frame heatmap
        for landmark in landmarks:
            frame_heatmap[landmark[1], landmark[0]] += 1  # Increase intensity at landmark position

        if heatmap is None:
            heatmap = frame_heatmap.copy()
        else:
            heatmap += frame_heatmap

        # Show the gesture recognition result
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        # Check if data_aux has fewer than 42 elements
        while len(data_aux) < 42:
            data_aux.extend([0.0, 0.0])  # Pad with zeros to reach 42 elements

        prediction = model.predict([np.asarray(data_aux[:42])])  # Use only the first 42 elements

        predicted_character = labels_dict[int(prediction[0])]


        cv2.putText(frame, predicted_character, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        recognized_gesture_data=[]
        print(f'landmark points for {predicted_character}')
        for landmark_id, landmark in enumerate(hand_landmarks.landmark):
            x = landmark.x * W
            y = landmark.y * H
            recognized_gesture_data.append({
            'Gesture': predicted_character,
            'X Coordinate': x,
            'Y Coordinate': y
            })

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Plotting the heatmap with improved visualization
plt.figure(figsize=(8, 6))
plt.title('Hand Landmark Heatmap')
plt.xlabel('Width')
plt.ylabel('Height')
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar(label='Landmark Density')
plt.tight_layout()
plt.show()

cap.release()
cv2.destroyAllWindows()

import pandas as pd
if recognized_gesture_data:
    tab=pd.DataFrame(recognized_gesture_data)
    print(tab)