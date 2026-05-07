import cv2 as cv
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 1)
cirlce_color = mp_drawing.DrawingSpec(color = (0,255,0), thickness = 3, circle_radius = 1)

def fingers_counting(landmarks):
    fingers = []

    lm = hand_landmarks.landmark
    fingers.append(lm[4].x > lm[3].x)

    for tip in [8, 12, 16, 20]:
        fingers.append(lm[tip].y < lm[tip - 2].y)

    return fingers.count(True)

video = cv.VideoCapture(0)
while True:
    isTrue, frame = video.read()
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    index_finger_pos = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_count = fingers_counting(hand_landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,cirlce_color)
            cv.putText(frame, f'Fingers: {finger_count}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv.imshow('Binary Count', frame)
    if cv.waitKey(10) & 0xFF==ord('q'):
        break


video.release()
cv.destroyAllWindows()
