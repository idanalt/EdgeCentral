# EdgeCentral — זיהוי גניבות/התנהגות חשודה בסופר/קיוסק מבוסס YOLO

מערכת לזיהוי אירועי גניבה פוטנציאליים בווידאו (מצלמות אבטחה) עבור סופרמרקט/קיוסק, המבוססת על:

1. **זיהוי אובייקטים (YOLO)** — אנשים, תיקים/תרמילים, פריטים.
2. **מעקב (tracking)** — שמירת זהות אובייקט לאורך פריימים (ByteTrack, מובנה ב-Ultralytics).
3. **שכבת היגיון התנהגותי (behavior heuristics)** — כיוון ש"גניבה" היא **התנהגות** ולא אובייקט בודד: הסתרת פריט בתיק/בגד, שהייה חריגה ליד מדף, תנועה מהירה של "אריזה" ללא קופה וכו'.
4. **ייצוא לקצה (edge)** — ONNX / TensorRT להרצה על מכשיר edge ייעודי (Jetson וכו').

> ⚠️ **חשוב — שימוש אחראי:** המערכת מייצרת **התראות לבדיקת אדם** (staff review), ולא קובעת אשמה אוטומטית. אין להשתמש בפלט כדי להאשים אדם ספציפי, לנעול דלתות אוטומטית, להזעיק משטרה אוטומטית, או לבצע זיהוי פנים/זיהוי אישי ללא ייעוץ משפטי מתאים (חוקי פרטיות, מצלמות אבטחה, הגנת הפרט). ודאו תאימות לחוק הישראלי (חוק הגנת הפרטיות, רשות להגנת הפרטיות) לפני פריסה בשטח.

## מבנה הפרויקט

```
configs/            הגדרות דאטהסט ואימון
scripts/            הורדת דאטהסט, הכנה, אימון, הערכה, ייצוא ל-edge
detection/          שכבת מעקב + היגיון התנהגותי + התראות
inference/          הרצת המודל בזמן אמת על מצלמה/RTSP/קובץ
edge/               הנחיות פריסה על מכשיר edge (Jetson וכו')
tests/              בדיקות יחידה להיגיון ההתנהגותי
data/               דאטהסטים (לא נכנס ל-git)
```

## התקנה

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## שלב 1: דאטהסט

אין באפשרותי לספק צילומי אבטחה אמיתיים. יש להשתמש בדאטהסט ציבורי מתויג (bounding boxes) בפורמט YOLO. אפשרות מומלצת: **Roboflow Universe** — יש שם כמה דאטהסטים ציבוריים בשם "shoplifting detection" / "theft detection" עם מחלקות כמו `person`, `bag`, `normal`, `shoplifting`.

1. הרשמה חינמית ל-https://roboflow.com וקבלת API key.
2. איתור דאטהסט מתאים תחת "Shoplifting Detection" ב-Roboflow Universe, ובחירת ה-workspace / project / version.
3. הגדרת משתני סביבה:

```bash
export ROBOFLOW_API_KEY=xxxx
export ROBOFLOW_WORKSPACE=<workspace-slug>
export ROBOFLOW_PROJECT=<project-slug>
export ROBOFLOW_VERSION=<version-number>
```

4. הורדה:

```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py
```

זה יוריד את הדאטהסט לתוך `data/raw/`, יאמת מבנה תקין (train/valid/test עם images+labels), ויכתוב `configs/dataset.yaml` סופי.

אם אין לכם עדיין דאטהסט ספציפי — אפשר להתחיל עם דאטהסט generic של `person` + `backpack`/`handbag` (למשל תת-קבוצה מ-COCO) כבייסליין, ולשפר בהמשך עם דאטהסט ייעודי.

### דוגמה: דאטהסט "Shoplifting" (weaponDetection, Roboflow)

https://universe.roboflow.com/weapondetection-jyvpf/shoplifting-sbnqg — 153 תמונות, Object Detection, 2 מחלקות: `Normal Behavior` / `Suspicious Behavior` (הקופסה עצמה מתייגת את ההתנהגות, לא "person"+"item" נפרדים). רישיון CC BY 4.0 — **נדרשת קרדיטציה** ל-workspace המקורי אם משתמשים בזה בפריסה.

⚠️ **153 תמונות זה דאטהסט קטן מאוד.** ה-mAP@50 של ~95% שמוצג בדף נמדד על ולידציה מאותו מאגר קטן/אותה חנות/מצלמה — לא אינדיקציה אמינה לביצועים בסופר/קיוסק אחר. השתמשו בזה כ-proof-of-concept, ותכננו לאסוף ולתייג צילומים מהאתר האמיתי שלכם לפני פריסה אמיתית.

מכיוון שהמחלקות כאן הן `Normal Behavior`/`Suspicious Behavior` ולא `person`/`item`, ה-heuristic של "פריט נעלם ליד אדם" לא רלוונטי לדאטהסט הזה. יש כאן heuristic מתאים — "sustained suspicious behavior" ב-`detection/behavior.py` — שמחכה שהמחלקה `Suspicious Behavior` תחזיק כמה שניות ברצף (לא רק פריים בודד רועש) לפני שמתריע. מריצים עם:

```bash
python -m inference.run_stream \
  --weights best.pt \
  --source 0 \
  --person-classes "Normal Behavior,Suspicious Behavior" \
  --suspicious-classes "Suspicious Behavior"
```

### פריסה במספר אתרים — מיזוג כמה דאטהסטים לאחד

אם היעד הוא לא אתר אחד אלא הרבה אתרים שונים, **אל תאמנו מודל נפרד לכל אתר** — זה לא ריאלי בקנה מידה. במקום זה, אספו דאטהסט **מגוון** (כמה דאטהסטים ציבוריים + כמה עשרות-מאות תמונות מכל אתר אמיתי שתתחילו לעבוד איתו, כולל סימולציות מבוימות ובהסכמה של "התנהגות חשודה") ומזגו לדאטהסט אחד לפני אימון. דאטהסטים שונים משתמשים לפעמים בשמות/אינדקסים שונים למחלקות (למשל `0`/`1` מול `Normal Behavior`/`Suspicious Behavior`) — `scripts/merge_datasets.py` ממפה את כולם לרשימת מחלקות אחידה:

1. הורידו כל דאטהסט בנפרד עם `scripts/download_dataset.py` (לכל פרויקט ב-Roboflow בנפרד — תוכלו לשנות את משתני הסביבה ולהריץ שוב, כל אחד ייכנס לתיקייה משלו תחת `data/raw/`).
2. העתיקו `configs/merge.example.yaml` ל-`configs/merge.yaml`, והגדירו `class_map` לכל מקור (איך למפות את שמות/אינדקסי המחלקות שלו ל-`unified_classes`).
3. הרצה:
   ```bash
   python scripts/merge_datasets.py
   ```
   זה כותב את הדאטהסט המאוחד ל-`data/merged/` וכותב `configs/dataset.yaml` חדש שמצביע אליו — ממשיכים ישר לאימון.

## שלב 2: אימון

```bash
python scripts/train.py --model yolo11n.pt --epochs 100 --imgsz 640
```

הגדרות ברירת מחדל נמצאות ב-`configs/train.yaml`. `yolo11n`/`yolov8n` (nano) מומלצים כברירת מחדל כי הם קלים מספיק להרצה על מכשיר edge בזמן אמת. **דרוש GPU לאימון** — לא זמין בסביבת ה-sandbox הזו; יש להריץ את שלב האימון במחשב/שרת עם GPU (או Colab / cloud GPU), ואז להעביר את משקלי `best.pt` חזרה למכשיר ה-edge.

## שלב 3: הערכה

```bash
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt
```

## שלב 4: ייצוא למכשיר Edge

```bash
python scripts/export_edge.py --weights runs/detect/train/weights/best.pt --format onnx
# ל-Jetson עם TensorRT:
python scripts/export_edge.py --weights runs/detect/train/weights/best.pt --format engine --half
```

ראו `edge/deploy_jetson.md` לפרטי פריסה על Jetson.

## שלב 5: הרצה בזמן אמת + התראות התנהגות

```bash
python -m inference.run_stream --weights best.onnx --source rtsp://<camera-ip>/stream --zones configs/zones.yaml
```

זה מריץ detection+tracking, מזין את התוצאות לשכבת ה-behavior heuristics (`detection/behavior.py`), ושומר התראות (תמונת snapshot + JSON) תחת `alerts/` לבדיקת צוות אבטחה.

## בדיקות

```bash
pytest tests/
```
