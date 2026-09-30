# מדריך דאטהסט

תיקייה זו (`data/raw/`, `data/yolo/`) לא נכנסת ל-git (ראו `.gitignore`) — הדאטהסט עצמו גדול מדי ולעיתים כפוף לרישוי של הצד השלישי שסיפק אותו.

## מקורות מומלצים

1. **Roboflow Universe** — חיפוש "shoplifting detection" / "theft detection" — יש כמה דאטהסטים ציבוריים עם bounding boxes בפורמט YOLO (מחלקות טיפוסיות: `person`, `normal`, `shoplifting`, `bag`). בדקו את הרישיון (license) של כל דאטהסט לפני שימוש מסחרי.
2. **דאטהסט משלכם** — הצילום הכי אמין הוא מהחנות/קיוסק עצמו. חשוב: יש לתייג עם קופסאות (bounding boxes) סביב אנשים ופריטים רלוונטיים, ולשמור פרטיות (לטשטש פנים של לקוחות שאינם רלוונטיים לתיוג אם החוק המקומי דורש זאת).

## פורמט צפוי (YOLOv8/YOLO11)

```
data/raw/<dataset-name>/
  data.yaml
  train/images/*.jpg   train/labels/*.txt
  valid/images/*.jpg   valid/labels/*.txt
  test/images/*.jpg    test/labels/*.txt   (אופציונלי)
```

הריצו `scripts/download_dataset.py` ואז `scripts/prepare_dataset.py` כדי לאמת ולהמיר.
