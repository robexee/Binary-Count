import cv2 as cv
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


video = cv.VideoCapture(0)
while True:
    isTrue, frame = video.read()
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape
    center_point = (w // 2, h // 2)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    index_finger_pos = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_pos = (int(index_finger_tip.x * w),
                                int(index_finger_tip.y * h))
            cv.circle(frame, index_finger_pos, 7, (255, 30, 30), -1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv.imshow('Binary Count', frame)
    if cv.waitKey(10) & 0xFF==ord('q'):
        break


video.release()
cv.destroyAllWindows()
