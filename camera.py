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
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    index_finger_pos = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv.imshow('Binary Count', frame)
    if cv.waitKey(10) & 0xFF==ord('q'):
        break


video.release()
cv.destroyAllWindows()
