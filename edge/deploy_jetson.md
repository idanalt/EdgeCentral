# פריסה על מכשיר Edge ייעודי (NVIDIA Jetson)

## דרישות
- Jetson Orin Nano / Xavier NX / AGX Orin עם JetPack (L4T) מותקן.
- Python 3.10+, `pip install ultralytics opencv-python`.
- כרטיס לא נדרש — Jetson כולל GPU מובנה.

## שלבים

1. **בניית מנוע TensorRT ספציפי למכשיר** — לא ניתן להעביר `.engine` שנבנה במחשב אחר; יש לבנות ישירות על ה-Jetson עצמו:

```bash
python scripts/export_edge.py --weights best.pt --format engine --half --imgsz 640
```

2. **בדיקת FPS בפועל**:

```bash
python -m inference.run_stream --weights best.engine --source rtsp://<camera>/stream --display
```

3. **טיפים לביצועים**:
   - השתמשו במודל `n` (nano) או `s` (small) בלבד — מודלים גדולים לא יעמדו בזמן אמת על Jetson.
   - `--half` (FP16) כמעט תמיד משתלם על Jetson; `--int8` דורש דאטת כיול (calibration) נפרדת.
   - הגבילו רזולוציית קלט (imgsz=640 או פחות) והורידו FPS מקור אם המצלמה תומכת (15 FPS מספיק בד"כ לזיהוי התנהגותי).
   - אם יש כמה מצלמות על אותו מכשיר — שקלו batch inference או תזמון round-robin בין מצלמות.

4. **חלופה: NVIDIA DeepStream** — לפריסה בקנה מידה (הרבה מצלמות על אותו Jetson), שקלו להריץ את מודל ה-YOLO המיוצא בתוך pipeline של DeepStream (GStreamer) במקום ה-Python loop הפשוט כאן — נותן ביצועים גבוהים משמעותית בריבוי סטרימים.

5. **תפעול שוטף**:
   - `alerts/events.jsonl` + snapshots נשמרים מקומית לבדיקת צוות אבטחה. חברו אותם ל-`WebhookSink` (ב-`detection/alerts.py`) כדי לשלוח לדשבורד מרכזי.
   - נטרו טמפרטורה/thermal throttling על Jetson בהרצה ממושכת (`tegrastats`).
   - שמרו מדיניות שמירת סרטונים/הצילומים תואמת לדרישות פרטיות (ראו README).
