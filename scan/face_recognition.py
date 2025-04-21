import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from mtcnn import MTCNN

# Load model dari folder model/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'face_recognition_mobilenetv2.h5')
model = load_model(MODEL_PATH)

# Fungsi prediksi wajah

detector = MTCNN()

def predict_face(frame):
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = detector.detect_faces(frame)

    status = "Unknown"
    # for (x, y, w, h) in faces:
    #     face = frame[y:y+h, x:x+w]
    #     face = cv2.resize(face, (224, 224))
    #     face = img_to_array(face)
    #     face = np.expand_dims(face, axis=0)
    #     face = face / 255.0
    for result in results:
        if result['confidence'] < 0.90:
            continue
    
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        prediction = model.predict(face)
        confidence = prediction.max(axis=1)[0] * 100
        # if prediction[0][0] > 0.5:
        if confidence > 30:
            status = "Blacklist"
        else:
            status = "Safe"
    return status
