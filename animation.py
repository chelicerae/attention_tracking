from attention_tracker import AttentionTracker
from fer import FER

import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import datetime as dt
import matplotlib.animation as animation

detector = FER()

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)


attention_tracker = AttentionTracker(
    config_path="./config.json", emotions_detector=detector
)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []


def animate(i, xs, ys):
    ret, frame = cap.read()

    res = attention_tracker.process_frame(frame=frame)
    metric = res["eyes_closed"]

    xs.append(dt.datetime.now().strftime("%H:%M:%S.%f"))
    ys.append(metric)

    xs = xs[-20:]
    ys = ys[-20:]

    ax.clear()
    ax.plot(xs, ys)

    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.30)
    plt.title("TMP102 Temperature over Time")
    plt.ylabel("Temperature (deg C)")


ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
plt.show()
cap.release()
