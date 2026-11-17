# 🧑‍🤝‍🧑 Face Gender Classifier

Predicts a person's gender ("male"/"female") from a face photo, using a deep learning model built with transfer learning.

## How it works

1. **OpenCV's Haar Cascade** detects the face region(s) in the input image.
2. Each detected face is cropped, resized to `224x224`, converted to RGB, and normalized.
3. A **Keras/TensorFlow CNN** (trained via transfer learning on a labeled face dataset) predicts the gender, and the result is drawn on the image with a confidence score.

## Tech stack

- **Python 3**
- **TensorFlow / Keras** — the classification model
- **OpenCV** — face detection and image handling
- **NumPy** — array preprocessing

## Installation

```bash
git clone https://github.com/<your-username>/gender-detection-face.git
cd gender-detection-face
pip install -r requirements.txt
```

## Usage

```bash
python predict.py path/to/photo.jpg
```

If no path is given, it falls back to the `IMAGE_PATH` set in the `CONFIG` section at the top of `predict.py`.

The model file `gender_classifier_face.h5` must be in the same folder as `predict.py` (already included in this repo).

## Note on the training pipeline

This repository includes the trained model and the **inference script only**. The original training script is not included; the model was trained with transfer learning on 224x224 RGB face crops with a binary (male/female) output.

## Demo

*(add a screenshot or short screen recording here showing a prediction on a sample photo)*

## Possible improvements

- Retrain with a proper (open, licensed) training script and dataset included in the repo, for full reproducibility
- Support batch prediction on a folder of images
- Replace Haar Cascade with a more accurate face detector (e.g. MTCNN, MediaPipe Face Detection)

## License

MIT
