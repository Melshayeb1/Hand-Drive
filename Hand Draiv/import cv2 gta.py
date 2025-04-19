import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import matplotlib.pyplot as plt
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
keyboard = Controller()

keys_state = {'w': False, 'a': False, 's': False, 'd': False}
current_direction = None  
prev_hand_x = None

def press_key(key):
    if not keys_state[key]:
        keyboard.press(key)
        keys_state[key] = True

def release_key(key):
    if keys_state[key]:
        keyboard.release(key)
        keys_state[key] = False

def release_all():
    for key in keys_state:
        release_key(key)

def draw_hand(landmarks):
    x = [lm.x for lm in landmarks]
    y = [lm.y for lm in landmarks]
    plt.cla()
    plt.scatter(x, y, c='red')
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.pause(0.001)

def detect_gesture_and_direction(landmarks):
    global prev_hand_x
    try:
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        wrist = landmarks[mp_hands.HandLandmark.WRIST]
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]

        is_open = index_tip.y < wrist.y and pinky_tip.y < wrist.y
        is_fist = index_tip.y > wrist.y and pinky_tip.y > wrist.y and abs(thumb_tip.x - thumb_ip.x) < 0.05

        current_x = (index_tip.x + pinky_tip.x) / 2
        direction = None

        if is_open and prev_hand_x is not None:
            movement = current_x - prev_hand_x
            if movement > 0.03:
                direction = 'right'
            elif movement < -0.03:
                direction = 'left'

        prev_hand_x = current_x
        return is_open, is_fist, direction
    except:
        return False, False, None

def main():
    global current_direction, prev_hand_x
    cap = cv2.VideoCapture(0)
    plt.ion()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_hand(hand_landmarks.landmark)

                is_open, is_fist, direction = detect_gesture_and_direction(hand_landmarks.landmark)

                if is_fist:
                    press_key('s')
                    release_key('w')
                    release_key('a')
                    release_key('d')
                    current_direction = None

                elif is_open:
                    press_key('w')
                    release_key('s')

                    if direction == 'right':
                        press_key('d')
                        release_key('a')
                        current_direction = 'right'
                    elif direction == 'left':
                        press_key('a')
                        release_key('d')
                        current_direction = 'left'
                    else:
                        if current_direction == 'right':
                            press_key('d')
                            release_key('a')
                        elif current_direction == 'left':
                            press_key('a')
                            release_key('d')
                        else:
                            release_key('a')
                            release_key('d')
                else:
                    release_all()
                    current_direction = None
        else:
            release_all()
            current_direction = None
            prev_hand_x = None

        cv2.imshow("Hand Control - GTA", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    release_all()
    plt.ioff()

if __name__ == "__main__":
    main()
