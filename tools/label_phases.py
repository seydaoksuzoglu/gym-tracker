"""

Önce bu şekilde labelling'ler yap.
python tools/label_phases.py <video2>.mp4 tests/fixtures/deadlift_002.json
Faz etiketleme tool'u.
Kullanim: python tools/label_phases.py <video_path> <output_json>

Soru: Lifter bar'ı yukarı çekip durduğunda, gövde tam dik mi?

Görsel kontrol listesi:

✅ Tam lockout (incomplete=yok):
Omuz, kalça, ayak bileği aynı dikey hatta
Bar baldırların önünde, kalçaya yapışık
Sırt dik, göğüs ileride
Sanki "askerce attention" durur gibi
❌ Incomplete (incomplete=VAR):
Hala hafif öne eğik bitirmiş
Kalça omuzun arkasında kalmış
Bar baldırdan ayrık, lifter sanki "neredeyse" bitirmiş ama tam değil
"Hocam göğsü ileri çıkarsana, kalçanı sıkıştır" denilecek görüntü
Pratik tanı: dikey bir hayali çizgi düşün. Omuz-kalça-bilek bu çizgide mi? Değilse → VAR.

uncontrolled_descent — descent fazında (3 bastıktan sonra, 4/5 basmadan önce) bak
Soru: Bar inerken kontrollü mü, "drop" mu?

Görsel kontrol listesi:

✅ Kontrollü (uncontrolled=yok):
İniş ~1-2 saniye sürüyor (çıkışla benzer süre)
Lifter bar'ı bilinçli aşağı kılavuzluyor
Bar zemine yumuşak değiyor, plate'ler "tak" sesi yerine yumuşak ses çıkarıyor
Lifter dengeli, sallanmıyor
❌ Uncontrolled (uncontrolled=VAR):
İniş 1 saniyeden kısa, "bıraktı" gibi
Bar gözle görülür şekilde düşüyor, indirilmiyor
Plate'ler "GÜÜMM" diye yere çarpıyor, zıplıyor olabilir
Lifter bar'ı yere yakınken "savuruyor" — eller bar'ı bırakmış gibi
Crossfit ekolünden ünlü "deadlift drop" — ağırlık çok ise normal kabul edilebilir ama V0 buna "uncontrolled" der

Kontroller:
  SPACE / D / -> : sonraki frame
  A / <-        : onceki frame
  Shift+SPACE   : 10 frame ileri
  Shift+A       : 10 frame geri
  1             : SETUP -> PULL
  2             : PULL -> LOCKOUT
  3             : LOCKOUT -> DESCENT
  4             : DESCENT -> PULL (ardisik rep)
  5             : DESCENT -> SETUP (son rep)
  BACKSPACE     : son eklenen etiketi sil
  S             : JSON kaydet
  Q             : cik

Akış
    Pencere açılır, ilk frame görünür
    Boşluk/D ile ileri, A ile geri — frame frame gezeriyorsun
    Bar yerden ayrıldığı an → 1 bas (setup → pull)
    Kullanıcı dikleştiği an → 2 bas (pull → lockout)
    İnişe geçtiği an → 3 bas
    Sonraki rep'in başı → 4 bas (veya son rep ise 5)
    Tüm geçişler işaretlendiğinde → S ile kaydet
    Q ile çık

[normal mod]
D ile ileri, A ile geri gez
1: setup→pull        (rep N başlar)
2: pull→lockout
3: lockout→descent
4 veya 5: descent→pull / descent→setup   (rep N KAPANIR)
    │
    └──► [otomatik labeling moduna geç]
            Ekranda overlay açılır:
            "REP N — HATA ETIKETLEME"
            [I] incomplete_lockout: yok / VAR  (toggle)
            [U] uncontrolled_descent: yok / VAR  (toggle)
            [1-5] quality: -
            ENTER kaydet  |  ESC iptal
            ENTER → rep_labels[]'a ekle, normal moda dön
            ESC → hem rep_label hem son transition iptal (undo)

"""
import argparse
import json
from pathlib import Path

import cv2


PHASE_KEYS = {
    ord("1"): ("setup", "pull"),
    ord("2"): ("pull", "lockout"),
    ord("3"): ("lockout", "descent"),
    ord("4"): ("descent", "pull"),
    ord("5"): ("descent", "setup"),
}
REP_CLOSE = {("descent", "pull"), ("descent", "setup")}

QUALITY_LABELS = {
    1: "perfect", 2: "good", 3: "needs_attention",
    4: "form_issue", 5: "severe",
}

# OpenCV key codes (Windows)
KEY_ENTER = 13
KEY_ESC = 27
KEY_BACKSPACE = 8


def save_json(output_path, video_name, variant, transitions, rep_labels):
    rep_count = sum(1 for t in transitions if (t["from"], t["to"]) in REP_CLOSE)
    data = {
        "video": video_name,
        "variant": variant,
        "rep_count": rep_count,
        "phase_transitions": transitions,
        "rep_labels": rep_labels,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_normal_overlay(display, frame_idx, total, ts_ms, transitions, rep_labels):
    info = f"frame={frame_idx}/{total-1}  ts={ts_ms}ms ({ts_ms/1000:.2f}s)"
    cv2.putText(display, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display, f"Transitions: {len(transitions)}  Reps etiketli: {len(rep_labels)}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if transitions:
        last = transitions[-1]
        cv2.putText(display,
                    f"son: {last['from']}->{last['to']} @ {last['ts_ms']}ms",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def render_labeling_overlay(display, rep_id, label):
    h, w = display.shape[:2]
    # Daha kucuk arka plan kutu (90px yukseklik)
    overlay = display.copy()
    cv2.rectangle(overlay, (5, 100), (w - 5, 195), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    cv2.putText(display, f"REP {rep_id} - ETIKETLEME", (12, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    il = "VAR" if label["incomplete_lockout"] else "yok"
    ud = "VAR" if label["uncontrolled_descent"] else "yok"
    q = label["quality"]
    q_text = f"{q}({QUALITY_LABELS[q]})" if q else "-"

    cv2.putText(display, f"[I] inc_lockout:{il}  [U] unc_desc:{ud}",
                (12, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(display, f"[1-5] quality:{q_text}",
                (12, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(display, "ENTER=kaydet  ESC=iptal",
                (12, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--variant", default="conventional",
                        choices=["conventional", "sumo", "romanian"])
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    transitions = []
    rep_labels = []

    labeling_mode = False
    pending_label = None  # dict, rep kapaninca olusturulur

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print("Video acilamadi.")
        cap.release()
        return
    last_good = frame

    while True:
        if not ok or frame is None:
            frame = last_good
        else:
            last_good = frame

        ts_ms = int(frame_idx * 1000 / fps)
        display = frame.copy()
        h, w = display.shape[:2]
        scale = min(1.0, 600 / w)
        if scale < 1.0:
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        if labeling_mode:
            render_labeling_overlay(display, pending_label["rep_id"], pending_label)
        else:
            render_normal_overlay(display, frame_idx, total, ts_ms,
                                  transitions, rep_labels)

        cv2.imshow("Phase + Error Labeler", display)
        key = cv2.waitKey(0) & 0xFF

        if labeling_mode:
            # ----- LABELING MODE -----
            if key == ord("i") or key == ord("I"):
                pending_label["incomplete_lockout"] = not pending_label["incomplete_lockout"]
            elif key == ord("u") or key == ord("U"):
                pending_label["uncontrolled_descent"] = not pending_label["uncontrolled_descent"]
            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                pending_label["quality"] = int(chr(key))
            elif key == KEY_ENTER:
                # commit
                pending_label["quality_label"] = (
                    QUALITY_LABELS[pending_label["quality"]]
                    if pending_label["quality"] else None
                )
                rep_labels.append(pending_label)
                save_json(output_path, video_path.name, args.variant,
                          transitions, rep_labels)
                print(f"  Rep {pending_label['rep_id']} kaydedildi: {pending_label}")
                pending_label = None
                labeling_mode = False
            elif key == KEY_ESC:
                # iptal et: son transition'i da geri al
                if transitions:
                    removed = transitions.pop()
                    print(f"  iptal + son transition silindi: {removed}")
                pending_label = None
                labeling_mode = False
                save_json(output_path, video_path.name, args.variant,
                          transitions, rep_labels)
            # diger tuslar mod icinde yok sayilir
            continue

        # ----- NORMAL MODE -----
        if key == ord("q"):
            break
        elif key in (ord("d"), ord(" "), 83):
            frame_idx = min(frame_idx + 1, total - 1)
        elif key in (ord("a"), 81):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord("D"):
            frame_idx = min(frame_idx + 10, total - 1)
        elif key == ord("A"):
            frame_idx = max(frame_idx - 10, 0)
        elif key in PHASE_KEYS:
            fr, to = PHASE_KEYS[key]
            transitions.append({"from": fr, "to": to, "ts_ms": ts_ms})
            print(f"  + {fr} -> {to} @ {ts_ms}ms (frame {frame_idx})")
            save_json(output_path, video_path.name, args.variant,
                      transitions, rep_labels)

            if (fr, to) in REP_CLOSE:
                # labeling moduna gec
                rep_id = len(rep_labels) + 1
                pending_label = {
                    "rep_id": rep_id,
                    "incomplete_lockout": False,
                    "uncontrolled_descent": False,
                    "quality": None,
                }
                labeling_mode = True
                print(f"  -> Rep {rep_id} kapandi, etiketleme modu")
        elif key == KEY_BACKSPACE:
            if transitions:
                removed = transitions.pop()
                print(f"  - silindi: {removed}")
                save_json(output_path, video_path.name, args.variant,
                          transitions, rep_labels)
        elif key == ord("s"):
            save_json(output_path, video_path.name, args.variant,
                      transitions, rep_labels)
            print(f"  Kaydedildi: {output_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
    save_json(output_path, video_path.name, args.variant,
              transitions, rep_labels)
    print(f"\nFinal kayit: {output_path}")


if __name__ == "__main__":
    main()
