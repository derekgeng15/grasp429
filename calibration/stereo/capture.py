# takes frames for 20s and creates sample imgs for stereo calibration
import cv2
import time
import numpy as np
import keyboard

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frames = []
t_end = time.time() + 5
while time.time() < t_end:
    cap.read()
print("starting...")
t_end = time.time() + 20
while time.time() < t_end:
    ret, frame = cap.read()
    if not ret:
        print("oops")
    frames.append(frame)
print(len(frames))
frames = [frames[i * int(len(frames) / 20)] for i in range(20)]

for i in range(len(frames)):
    cv2.imwrite(f"imgs/frames/{i}.JPG", frames[i])
    left = frames[i][:, :int(frames[i].shape[1]/2), :]
    right = frames[i][:, int(frames[i].shape[1]/2):, :]
    cv2.imwrite(f"imgs/LEFT/{i}.JPG", left)
    cv2.imwrite(f"imgs/RIGHT/{i}.JPG", right)
    
cap.release()
