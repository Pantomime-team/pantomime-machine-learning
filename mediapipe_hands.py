import cv2
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

hands = mpHands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tracked_hands = hands.process(frameRGB)

    if tracked_hands.multi_hand_landmarks:
        for landmark in tracked_hands.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, landmark, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Mediapipe test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
