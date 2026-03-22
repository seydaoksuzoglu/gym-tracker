# Models

Bu klasördeki model dosyaları büyük binary dosyalar olduğu için repoya dahil edilmemiştir.

## Gerekli Dosyalar

### MediaPipe Pose Landmarker
- **Dosya:** `pose_landmarker_full.task`
- **İndirme:** [Google MediaPipe Model Card](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models)
- **Direkt link:**
  ```
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
  ```
- **Boyut:** ~5.5 MB

**İndirme komutu:**
```bash
curl -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

### YOLO Pose (opsiyonel)
- **Dosya:** `yolo26m-pose.pt` (veya `yolo11m-pose.pt`)
- **İndirme:** [Ultralytics](https://docs.ultralytics.com/models/) — `YOLO("yolo26m-pose.pt")` çağrısı ile otomatik indirilir.
