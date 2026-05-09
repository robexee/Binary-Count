import cv2 as cv
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 1)
cirlce_color = mp_drawing.DrawingSpec(color = (0,255,0), thickness = 3, circle_radius = 1)


def resize_camera(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    resized_frame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    return resized_frame


def binary_count(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark

    if hand_label == 'Right':
        thumb = lm[4].x < lm[3].x
    else:
        thumb = lm[4].x > lm[3].x
    
    fingers = [
        thumb,
        lm[8].y < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y
    ]
    decimal_value = sum(state << i for i, state in enumerate(fingers))
    binary_value = "".join(str(int(state)) for state in fingers[::-1])
    return decimal_value, binary_value


video = cv.VideoCapture(0)
while True:
    isTrue, frame = video.read()
    frame = cv.flip(frame, 1)
    frame = resize_camera(frame, 2)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            
            decimal_val, binary_str = binary_count(hand_landmarks, hand_label)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, cirlce_color)

            cv.putText(frame, f"Decimal Number: {decimal_val}", (10, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv.putText(frame, f"Binary: {binary_str}", (frame.shape[1] - 250, 46), cv.FONT_HERSHEY_COMPLEX,1 , (255,255,255), 1)

    cv.imshow('Binary Count', frame)
    if cv.waitKey(10) & 0xFF==ord('q'):
        break

video.release()
cv.destroyAllWindows()
