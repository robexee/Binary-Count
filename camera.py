import cv2 as cv
import mediapipe as mp
from collections import deque, Counter
import time

font = cv.FONT_HERSHEY_COMPLEX
class HandBinaryTracker:

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.5, tracking_conf:  float = 0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence = detection_conf,
            min_tracking_confidence = tracking_conf,
            max_num_hands = max_hands
        )
        self.circle_color = self.mp_drawing.DrawingSpec(color=(0,255,0), thickness = 3, circle_radius = 1)
        self.buffer = deque(maxlen=5)

    def get_binary_count(self, hand_landmarks, hand_label):
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
        binary_string = "".join(str(int(state)) for state in fingers[::-1])

        self.buffer.append((decimal_value, binary_string))
        most_freq = Counter(self.buffer).most_common(1)[0][0]

        return most_freq[0], most_freq[1]
    
    def draw(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.circle_color
        )


def resize_camera(frame, scale: float = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
    

def main():
    prev_frame_time = 0
    new_frame_time = 0
    video = cv.VideoCapture(0)
    tracker = HandBinaryTracker()

    while True:
        isTrue, frame = video.read()
        if not isTrue:
            break

        frame = cv.flip(frame, 1)
        frame = resize_camera(frame, 2)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150,40), font, 1, (255,255,255), 1)
        results = tracker.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip (results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label

                decimal_val, binary_str = tracker.get_binary_count(hand_landmarks, hand_label)
                tracker.draw(frame, hand_landmarks)

                cv.putText(frame, f"Decimal: {decimal_val}", (10,40), font, 1, (255,255,255), 1)
                cv.putText(frame, f"Binary: {binary_str}", (10, 70), font, 1, (255,255,255), 1)
        else:
            tracker.buffer.clear()
        
        cv.imshow('Binary Finger Counter', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    
    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()