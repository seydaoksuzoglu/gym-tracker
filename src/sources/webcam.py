import time
import cv2

def webcam_frames(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Webcam açılamadı. Kamera index'i 0/1/2 deneyebilirsin.")

    last_ts = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts = int(time.time() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            yield frame_bgr, ts
    finally:
        cap.release()