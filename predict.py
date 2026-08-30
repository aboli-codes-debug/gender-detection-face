"""
تشخیص جنسیت از روی تصویر چهره (Face Gender Classifier)
========================================================

یک مدل شبکه‌ی عصبی (Transfer Learning, ورودی 224x224) که روی چهره‌ی
تشخیص‌داده‌شده در عکس اجرا می‌شود و «مرد» یا «زن» را با درصد اطمینان
پیش‌بینی می‌کند.

اجرا:
    python predict.py path/to/image.jpg

اگر مسیر عکس داده نشود، از IMAGE_PATH پیش‌فرض در بخش CONFIG استفاده می‌شود.

نکته: این اسکریپت فقط برای پیش‌بینی (inference) با مدل از پیش آموزش‌دیده
است؛ کد آموزش مدل (training) در این پروژه شامل نمی‌شود.
"""

import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================================================
#                       CONFIG (قابل تغییر)
# =========================================================

MODEL_PATH = "gender_classifier_face.h5"
IMAGE_PATH = "test.jpg"          # اگر آرگومان خط فرمان داده نشود، از این استفاده می‌شود
IMAGE_SIZE = (224, 224)          # باید با اندازه‌ی ورودی مدل در زمان آموزش یکی باشد
CONFIDENCE_LABELS = ("زن", "مرد")  # (برچسب کلاس ۰, برچسب کلاس ۱)

FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

# =========================================================
#                       END CONFIG
# =========================================================


def load_face_detector():
    """تشخیص‌دهنده‌ی چهره‌ی Haar Cascade که به‌صورت پیش‌فرض همراه OpenCV است."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"فایل Haar Cascade پیدا نشد یا لود نشد: {cascade_path}")
    return detector


def preprocess_face(face_bgr):
    """ناحیه‌ی صورت (BGR, هر اندازه) را به همان فرمتی که مدل موقع آموزش
    دیده تبدیل می‌کند: RGB، نرمال‌شده بین 0 و 1، با یک بعد اضافه برای batch."""
    face = cv2.resize(face_bgr, IMAGE_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


def predict_gender(image_path: str, model, face_detector) -> None:
    """چهره‌(ها) را در عکس پیدا کرده، جنسیت هرکدام را پیش‌بینی می‌کند، و
    نتیجه را هم در ترمینال چاپ و هم روی تصویر رسم می‌کند."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"عکس پیدا نشد یا مسیر اشتباه است: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBORS
    )

    if len(faces) == 0:
        print("هیچ صورتی در عکس پیدا نشد")
        return

    for (x, y, w, h) in faces:
        face_input = preprocess_face(img[y:y + h, x:x + w])

        pred = model.predict(face_input, verbose=0)[0][0]
        label = CONFIDENCE_LABELS[1] if pred > 0.5 else CONFIDENCE_LABELS[0]
        confidence = pred if pred > 0.5 else 1 - pred

        print(f"نتیجه: {label}  (اطمینان: {confidence * 100:.1f}%)")

        cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
        cv2.putText(
            img, f"{label} {confidence * 100:.1f}%", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2,
        )

    cv2.imshow("Result - press any key to close", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        model = load_model(MODEL_PATH)
    except (OSError, IOError) as e:
        print(f"خطا در بارگذاری مدل ({MODEL_PATH}): {e}")
        sys.exit(1)

    face_detector = load_face_detector()

    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    predict_gender(image_path, model, face_detector)
