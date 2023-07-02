from attention_tracker import AttentionTracker
from fer import FER

import cv2
import mediapipe as mp
import json

detector = FER()

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)


attention_tracker = AttentionTracker(
    config_path="./config.json", emotions_detector=detector
)

with open("fake_happy3.json", "a") as face_data_json:
    while True:
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        if not ret:
            break

        res = attention_tracker.process_frame(frame=frame)
        # json.dump(fp=face_data_json, obj=res)

        face_data_json.write(json.dumps(res))
        face_data_json.write("\n")

        print(res)

        # cv2.putText(
        #     img=frame,
        #     text=res["head_turn"],
        #     org=(20, 50),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=1.5,
        #     color=(0, 0, 255),
        #     thickness=2,
        # )

        # if res["eyes_closed"] > 5.3:
        #     eye_label = "Eyes closed"
        # else:
        #     eye_label = "Eyes opened"

        # cv2.putText(
        #     img=frame,
        #     text=eye_label,
        #     org=(20, 100),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=1.5,
        #     color=(0, 0, 255),
        #     thickness=2,
        # )

        # if res["is_face_detected"]:
        #     max_emotion = max(res["emotions"], key=res["emotions"].get)
        #     cv2.putText(
        #         img=frame,
        #         text=f"Emotion: {max_emotion}",
        #         org=(20, 150),
        #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=1.5,
        #         color=(0, 0, 255),
        #         thickness=2,
        #     )

        # cv2.imshow("Head Pose Estimation", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
