import os
import json
import cv2
import argparse
import matplotlib.pyplot as plt


# --------------------------------------------------
# JSON OKUMA
# --------------------------------------------------
def load_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# KLASÖR OLUŞTURMA
# --------------------------------------------------
def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------
# GÜVENLİ FRAME OKUMA
# --------------------------------------------------
def safe_read_frame(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


# --------------------------------------------------
# REP JSON İÇİNDEKİ RAW FRAME KAYDINI BULMA
# --------------------------------------------------
def get_raw_frame_record(rep_data, frame_idx: int):
    for frame_record in rep_data.get("raw_rep_frames", []):
        if frame_record.get("frame_index") == frame_idx:
            return frame_record
    return None


# --------------------------------------------------
# MEDIAPIPE POSE CONNECTIONS
# --------------------------------------------------
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32)
]


# --------------------------------------------------
# FRAME ÜZERİNE LANDMARK ÇİZME
# --------------------------------------------------
def draw_pose_on_frame(frame_bgr, frame_record):
    if frame_record is None:
        return frame_bgr

    image_landmarks = frame_record.get("image_landmarks", [])
    if not image_landmarks:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    for a, b in POSE_CONNECTIONS:
        if a < len(image_landmarks) and b < len(image_landmarks):
            lm_a = image_landmarks[a]
            lm_b = image_landmarks[b]

            xa = int(lm_a["x"] * w)
            ya = int(lm_a["y"] * h)
            xb = int(lm_b["x"] * w)
            yb = int(lm_b["y"] * h)

            cv2.line(annotated, (xa, ya), (xb, yb), (0, 255, 0), 2)

    for lm in image_landmarks:
        x = int(lm["x"] * w)
        y = int(lm["y"] * h)
        cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)

    return annotated


# --------------------------------------------------
# FEATURE SERİLERİNİ ÇIKARMA
# --------------------------------------------------
def extract_series(frame_features, key):
    xs = []
    ys = []

    for item in frame_features:
        xs.append(item["frame_index"])
        ys.append(item.get(key))

    return xs, ys


# --------------------------------------------------
# LABEL METNİ OLUŞTURMA
# --------------------------------------------------
def format_labels(labels: dict):
    active = [k for k, v in labels.items() if v == 1]
    if not active:
        return "correct"
    return ", ".join(active)


# --------------------------------------------------
# DOSYA ADI GÜVENLİLEŞTİRME
# --------------------------------------------------
def sanitize_filename(text: str):
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        text = text.replace(ch, "_")
    return text


# --------------------------------------------------
# GÖRSEL KAYDETME
# --------------------------------------------------
def save_rgb_image(rgb_image, save_path):
    """
    Matplotlib için RGB tuttuğumuz resmi OpenCV ile kaydederken tekrar BGR'ye çeviriyoruz.
    """
    if rgb_image is None:
        return
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr)


# --------------------------------------------------
# TEK REP GÖRSELLEŞTİRME
# --------------------------------------------------
def visualize_rep(rep_json_path: str, videos_root: str, save_dir: str = None):
    rep_data = load_json(rep_json_path)

    video_rel_path = rep_data["video_path"]
    video_path = os.path.join(videos_root, video_rel_path)

    if not os.path.isfile(video_path):
        print(f"[HATA] Video bulunamadı: {video_path}")
        return

    start_frame = rep_data["start_frame"]
    bottom_frame = rep_data["bottom_frame"]
    end_frame = rep_data["end_frame"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        return

    start_img = safe_read_frame(cap, start_frame)
    bottom_img = safe_read_frame(cap, bottom_frame) if bottom_frame is not None else None
    end_img = safe_read_frame(cap, end_frame)
    cap.release()

    start_raw = get_raw_frame_record(rep_data, start_frame)
    bottom_raw = get_raw_frame_record(rep_data, bottom_frame) if bottom_frame is not None else None
    end_raw = get_raw_frame_record(rep_data, end_frame)

    if start_img is not None:
        start_img = draw_pose_on_frame(start_img, start_raw)
        start_img = cv2.cvtColor(start_img, cv2.COLOR_BGR2RGB)

    if bottom_img is not None:
        bottom_img = draw_pose_on_frame(bottom_img, bottom_raw)
        bottom_img = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2RGB)

    if end_img is not None:
        end_img = draw_pose_on_frame(end_img, end_raw)
        end_img = cv2.cvtColor(end_img, cv2.COLOR_BGR2RGB)

    frame_features = rep_data.get("frame_features", [])

    x_knee_l, y_knee_l = extract_series(frame_features, "left_knee_angle")
    x_knee_r, y_knee_r = extract_series(frame_features, "right_knee_angle")

    x_hip_l, y_hip_l = extract_series(frame_features, "left_hip_angle")
    x_hip_r, y_hip_r = extract_series(frame_features, "right_hip_angle")

    x_torso_l, y_torso_l = extract_series(frame_features, "left_torso_lean_deg")
    x_torso_r, y_torso_r = extract_series(frame_features, "right_torso_lean_deg")

    labels_text = format_labels(rep_data.get("labels", {}))
    summary = rep_data.get("summary_features", {})

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    if start_img is not None:
        ax1.imshow(start_img)
    ax1.set_title(f"START\nframe={start_frame}")
    ax1.axis("off")

    if bottom_img is not None:
        ax2.imshow(bottom_img)
    ax2.set_title(f"BOTTOM\nframe={bottom_frame}")
    ax2.axis("off")

    if end_img is not None:
        ax3.imshow(end_img)
    ax3.set_title(f"END\nframe={end_frame}")
    ax3.axis("off")

    ax4.plot(x_knee_l, y_knee_l, label="Left Knee Angle")
    ax4.plot(x_knee_r, y_knee_r, label="Right Knee Angle")
    ax4.axvline(start_frame, linestyle="--", label="Start")
    if bottom_frame is not None:
        ax4.axvline(bottom_frame, linestyle="--", label="Bottom")
    ax4.axvline(end_frame, linestyle="--", label="End")
    ax4.set_title("Knee Angles")
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Degree")
    ax4.legend()

    ax5.plot(x_hip_l, y_hip_l, label="Left Hip Angle")
    ax5.plot(x_hip_r, y_hip_r, label="Right Hip Angle")
    ax5.axvline(start_frame, linestyle="--", label="Start")
    if bottom_frame is not None:
        ax5.axvline(bottom_frame, linestyle="--", label="Bottom")
    ax5.axvline(end_frame, linestyle="--", label="End")
    ax5.set_title("Hip Angles")
    ax5.set_xlabel("Frame")
    ax5.set_ylabel("Degree")
    ax5.legend()

    ax6.plot(x_torso_l, y_torso_l, label="Left Torso Lean")
    ax6.plot(x_torso_r, y_torso_r, label="Right Torso Lean")
    ax6.axvline(start_frame, linestyle="--", label="Start")
    if bottom_frame is not None:
        ax6.axvline(bottom_frame, linestyle="--", label="Bottom")
    ax6.axvline(end_frame, linestyle="--", label="End")
    ax6.set_title("Torso Lean")
    ax6.set_xlabel("Frame")
    ax6.set_ylabel("Degree")
    ax6.legend()

    fig.suptitle(
        f"Video: {rep_data['video_name']} | Rep ID: {rep_data['rep_id']} | Labels: {labels_text}\n"
        f"Min L Knee: {summary.get('min_left_knee_angle')} | "
        f"Min R Knee: {summary.get('min_right_knee_angle')} | "
        f"End L Knee: {summary.get('end_left_knee_angle')} | "
        f"End R Knee: {summary.get('end_right_knee_angle')}",
        fontsize=12
    )

    plt.tight_layout()

    # --------------------------------------------------
    # KAYDETME BLOĞU
    # --------------------------------------------------
    if save_dir is not None:
        ensure_dir(save_dir)

        video_name = sanitize_filename(os.path.splitext(rep_data["video_name"])[0])
        rep_id = rep_data["rep_id"]
        base_name = f"{video_name}_rep_{rep_id}"

        figure_path = os.path.join(save_dir, base_name + "_summary.png")
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")

        start_path = os.path.join(save_dir, base_name + "_start.png")
        end_path = os.path.join(save_dir, base_name + "_end.png")
        bottom_path = os.path.join(save_dir, base_name + "_bottom.png")

        if start_img is not None:
            save_rgb_image(start_img, start_path)
        if bottom_img is not None:
            save_rgb_image(bottom_img, bottom_path)
        if end_img is not None:
            save_rgb_image(end_img, end_path)

        print(f"[KAYDEDİLDİ] Figure: {figure_path}")
        if start_img is not None:
            print(f"[KAYDEDİLDİ] Start : {start_path}")
        if bottom_img is not None:
            print(f"[KAYDEDİLDİ] Bottom: {bottom_path}")
        if end_img is not None:
            print(f"[KAYDEDİLDİ] End   : {end_path}")

    plt.show()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tek bir rep JSON dosyasını görselleştirir ve isterse klasöre kaydeder.")

    parser.add_argument(
        "--rep_json",
        type=str,
        required=True,
        help="Rep JSON dosyasının tam yolu"
    )

    parser.add_argument(
        "--videos_root",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\videos\squat",
        help="Videoların kök klasörü"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\visualizations\squat",
        help="Görsellerin kaydedileceği klasör"
    )

    args = parser.parse_args()

    visualize_rep(args.rep_json, args.videos_root, args.save_dir)