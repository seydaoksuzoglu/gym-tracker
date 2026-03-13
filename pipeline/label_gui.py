import cv2
import os
import json
import glob

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def get_video_files(folder_path):
    search_pattern = os.path.join(folder_path, "**", "*.mp4")
    return glob.glob(search_pattern, recursive=True)

def get_output_json_path(video_path, input_root, output_root):
    relative_video_path = os.path.relpath(video_path, input_root)
    relative_without_ext = os.path.splitext(relative_video_path)[0]
    return os.path.join(output_root, relative_without_ext + ".json")

def load_or_create_annotation(json_path, video_path, folder_path, fps, total_frames, exercise="squat"):
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "reps" not in data:
            data["reps"] = []
        return data

    return {
        "video_name": os.path.basename(video_path),
        "video_path": os.path.relpath(video_path, folder_path),
        "exercise": exercise,
        "fps": fps,
        "total_frames": total_frames,
        "reps": []
    }

def save_annotation(data, json_path):
    ensure_dir(os.path.dirname(json_path))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sec_from_frame(frame_idx, fps):
    if fps <= 0:
        return 0.0
    return frame_idx / fps

def reset_current_rep():
    return {
        "start_frame": None,
        "bottom_frame": None,
        "end_frame": None,
        "labels": {
            "correct": 0,
            "half_depth": 0,
            "knee_error": 0,
            "incomplete_lockout": 0
        }
    }
def normalize_labels(labels):
    # Eğer correct seçildiyse diğer hatalar kapansın
    if labels["correct"] == 1:
        labels["half_depth"] = 0
        labels["knee_error"] = 0
        labels["incomplete_lockout"] = 0
    return labels

def safe_read_frame(cap, frame_idx, total_frames):
    # Frame indeksini güvenli aralıkta tut
    frame_idx = max(0, min(frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret or frame is None:
        return False, None

    return True, frame

def draw_info_panel(display_frame, idx, total_videos, current_frame_idx, total_frames, fps, is_paused, reps, current_rep):
    current_time_sec = sec_from_frame(current_frame_idx, fps)

    height, width = display_frame.shape[:2]
    max_height = 700
    if height > max_height:
        scale = max_height / height
        display_frame = cv2.resize(display_frame, (int(width * scale), int(height * scale)))

    lines = [
        f"Video: {idx + 1}/{total_videos}",
        f"Frame: {current_frame_idx}/{total_frames - 1}",
        f"Sure: {current_time_sec:.2f} sn",
        f"Kayitli Rep: {len(reps)}",
        f"Pause: {'Evet' if is_paused else 'Hayir'}",
        f"Start: {current_rep['start_frame']}",
        f"Bottom: {current_rep['bottom_frame']}",
        f"End: {current_rep['end_frame']}",
        f"Labels: {current_rep['labels']}"
    ]

    y = 30
    for line in lines:
        cv2.putText(display_frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 28

    help_text = "SPACE:Play/Pause | J/L:Geri/Ileri | A:Start | B:Bottom | D:End | 1/2/3/4:Label | ENTER:Save | Z:Undo | X:Cancel | N:Next | Q:Quit"
    cv2.putText(display_frame, help_text, (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    if is_paused:
        cv2.putText(display_frame, "DURAKLATILDI", (10, display_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return display_frame

def process_folder(folder_path, output_root):
    video_files = get_video_files(folder_path)

    if not video_files:
        print(f"HATA: '{folder_path}' klasöründe .mp4 video bulunamadı.")
        return

    print(f"\nToplam {len(video_files)} video bulundu.")
    print("\nKısayollar:")
    print("[SPACE]  : Oynat / Durdur")
    print("[J / L]  : Duraklatılmışken 1 frame geri / ileri")
    print("[A]      : Rep başlangıcı")
    print("[B]      : Rep dip noktası")
    print("[D]      : Rep bitişi")
    print("[1]      : Correct")
    print("[2]      : Half Depth")
    print("[3]      : Knee Error")
    print("[4]      : Incomplete Lockout")
    print("[ENTER]  : Rep kaydet")
    print("[Z]      : Son repi geri al")
    print("[X]      : Aktif repi iptal et")
    print("[N]      : Sonraki video")
    print("[Q]      : Çık ve kaydet\n")

    quit_program = False

    for idx, video_path in enumerate(video_files):
        print("\n" + "=" * 60)
        print(f"🎥 VIDEO {idx+1}/{len(video_files)}")
        print(f"Dosya: {video_path}")
        print("=" * 60)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Uyarı: {video_path} açılamadı, atlanıyor.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wait_ms = max(1, int(1000 / fps))

        json_path = get_output_json_path(video_path, folder_path, output_root)
        annotation = load_or_create_annotation(json_path, video_path, folder_path, fps, total_frames, exercise="squat")
        reps = annotation["reps"]

        print(f"JSON: {json_path}")
        print(f"Mevcut rep sayısı: {len(reps)}")

        current_frame_idx = 0
        ret, frame = safe_read_frame(cap, current_frame_idx, total_frames)
        if not ret:
            print("İlk frame okunamadı, video atlanıyor.")
            cap.release()
            continue

        is_paused = True
        skip_video = False
        current_rep = reset_current_rep()

        while cap.isOpened():
            if frame is None:
                print(f"HATA: frame okunamadı. Video={video_path}, frame={current_frame_idx}")
                break
            display_frame = draw_info_panel(
                frame.copy(),
                idx,
                len(video_files),
                current_frame_idx,
                total_frames,
                fps,
                is_paused,
                reps,
                current_rep
            )

            cv2.imshow("Video Etiketleme - Tek Video Tek JSON", display_frame)
            key = cv2.waitKey(wait_ms if not is_paused else 0) & 0xFF

            if key == ord('q'):
                save_annotation(annotation, json_path)
                quit_program = True
                break

            elif key == ord('n'):
                save_annotation(annotation, json_path)
                skip_video = True
                break

            elif key == ord(' '):
                is_paused = not is_paused

            elif key == ord('j') and is_paused:
                new_idx = max(0, current_frame_idx - 1)
                ret, new_frame = safe_read_frame(cap, new_idx, total_frames)
                if ret:
                    current_frame_idx = new_idx
                    frame = new_frame
                else:
                    print("Önceki frame okunamadı.")

            elif key == ord('l') and is_paused:
                new_idx = min(total_frames - 1, current_frame_idx + 1)
                ret, new_frame = safe_read_frame(cap, new_idx, total_frames)
                if ret:
                    current_frame_idx = new_idx
                    frame = new_frame
                else:
                    print("Son okunabilir frame'e gelindi.")

            elif key == ord('a'):
                current_rep = reset_current_rep()
                current_rep["start_frame"] = current_frame_idx
                print(f"Rep başlangıcı: frame {current_frame_idx}")

            elif key == ord('b'):
                if current_rep["start_frame"] is not None:
                    current_rep["bottom_frame"] = current_frame_idx
                    print(f"Bottom frame: {current_frame_idx}")
                else:
                    print("Önce start frame seç.")

            elif key == ord('d'):
                if current_rep["start_frame"] is not None:
                    current_rep["end_frame"] = current_frame_idx
                    print(f"Rep bitişi: frame {current_frame_idx}")
                else:
                    print("Önce start frame seç.")

            elif key == ord('1'):
                current_rep["labels"]["correct"] = 1
                current_rep["labels"]["half_depth"] = 0
                current_rep["labels"]["knee_error"] = 0

            elif key == ord('2'):
                current_rep["labels"]["half_depth"] = 1 - current_rep["labels"]["half_depth"]
                if current_rep["labels"]["half_depth"] == 1:
                    current_rep["labels"]["correct"] = 0

            elif key == ord('3'):
                current_rep["labels"]["knee_error"] = 1 - current_rep["labels"]["knee_error"]
                if current_rep["labels"]["knee_error"] == 1:
                    current_rep["labels"]["correct"] = 0
            elif key == ord('4'):
                current_rep["labels"]["incomplete_lockout"] = 1 - current_rep["labels"]["incomplete_lockout"]
                if current_rep["labels"]["incomplete_lockout"] == 1:
                    current_rep["labels"]["correct"] = 0
                
            elif key == ord('x'):
                current_rep = reset_current_rep()
                print("Aktif rep iptal edildi.")

            elif key == ord('z'):
                if reps:
                    removed = reps.pop()
                    save_annotation(annotation, json_path)
                    print(f"Son rep silindi: rep_id={removed['rep_id']}")
                else:
                    print("Silinecek rep yok.")

            elif key == 13:  # Enter
                if current_rep["start_frame"] is None:
                    print("HATA: Start frame seçilmedi.")
                elif current_rep["end_frame"] is None:
                    print("HATA: End frame seçilmedi.")
                elif current_rep["end_frame"] <= current_rep["start_frame"]:
                    print("HATA: End frame, start frame'den büyük olmalı.")
                else:
                    current_rep["labels"] = normalize_labels(current_rep["labels"])

                    rep_id = len(reps) + 1
                    rep_data = {
                        "rep_id": rep_id,
                        "start_frame": current_rep["start_frame"],
                        "bottom_frame": current_rep["bottom_frame"],
                        "end_frame": current_rep["end_frame"],
                        "start_sec": round(sec_from_frame(current_rep["start_frame"], fps), 3),
                        "bottom_sec": round(sec_from_frame(current_rep["bottom_frame"], fps), 3) if current_rep["bottom_frame"] is not None else None,
                        "end_sec": round(sec_from_frame(current_rep["end_frame"], fps), 3),
                        "labels": current_rep["labels"]
                    }

                    reps.append(rep_data)
                    save_annotation(annotation, json_path)
                    print(f"Rep kaydedildi: rep_id={rep_id}")
                    current_rep = reset_current_rep()

            if not is_paused:
                if current_frame_idx >= total_frames - 1:
                    print("Video sonuna gelindi.")
                    save_annotation(annotation, json_path)
                    break

                new_idx = current_frame_idx + 1
                ret, new_frame = safe_read_frame(cap, new_idx, total_frames)

                if not ret:
                    print(f"Sonraki frame okunamadı. Son okunabilir frame: {current_frame_idx}")
                    save_annotation(annotation, json_path)
                    break

                current_frame_idx = new_idx
                frame = new_frame

        cap.release()

        if skip_video:
            print("Sonraki videoya geçiliyor.")

        if quit_program:
            print("Program sonlandırıldı.")
            break

    cv2.destroyAllWindows()
    print("\nTüm mevcut veriler kaydedildi.")

if __name__ == "__main__":
    klasor_yolu = r"C:\Users\Seyda\Desktop\Projects\GymTracker\videos\squat"
    output_root = r"C:\Users\Seyda\Desktop\Projects\GymTracker\video_labels\squat"

    ensure_dir(klasor_yolu)
    ensure_dir(output_root)

    video_files = get_video_files(klasor_yolu)
    if not video_files:
        print(f"Lütfen videolarınızı '{klasor_yolu}' klasörüne koyun ve tekrar çalıştırın.")
    else:
        process_folder(klasor_yolu, output_root)