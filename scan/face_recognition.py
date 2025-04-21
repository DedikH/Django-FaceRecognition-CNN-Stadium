import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from mtcnn import MTCNN
import joblib
from sklearn.preprocessing import LabelEncoder
from .get_pelanggar_labeling import get_pelanggar_by_label 

# Load model dari folder model/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'face_recognition_mobilenetv2.h5')
model = load_model(MODEL_PATH)

LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'label_encoder.pkl')
label_encoder = joblib.load(LABEL_ENCODER_PATH)
# Fungsi prediksi wajah

detector = MTCNN()

def predict_face(frame):
    results = detector.detect_faces(frame)

    if len(results) == 0:
        return "Safe", "No Face Detected"

    for result in results:
        if result['confidence'] < 0.90:
            continue

        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)

        face = frame[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (224, 224))
        except Exception as e:
            print("Resize error:", e)
            continue

        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        prediction = model.predict(face)
        confidence = prediction.max(axis=1)[0] * 100
        label_predicted = np.argmax(prediction)
        label_name = label_encoder.inverse_transform([label_predicted])[0]

        data_pelanggar = get_pelanggar_by_label(label_name)

        if confidence > 30 and data_pelanggar:
            return "Blacklist", label_name
        else:
            return "Safe", label_name

    return "Safe", "No Confident Face"