import cv2

def video_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0  # fallback

    step_ms = 1000.0 / fps
    frame_idx = 0
    last_ts = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts = int(frame_idx * step_ms)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            frame_idx += 1
            yield frame_bgr, ts
    finally:
        cap.release()