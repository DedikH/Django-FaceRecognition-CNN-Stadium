import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from mtcnn import MTCNN
import time
import base64
from io import BytesIO
from PIL import Image

# Load model CNN
model = tf.keras.models.load_model('face_recognition_mobilenetv2.h5')
labels = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma', 'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Courtney Cox', 'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman', 'Jessica Alba'
]

# Inisialisasi kamera
cap = cv2.VideoCapture(1)
detector = MTCNN()

def index(request):
    return render(request, 'detector/index.html')

# Kamera streaming
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def snapshot_api(request):
    success, frame = cap.read()
    if not success:
        return JsonResponse({'results': []})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    results = []
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb_frame[y:y+h, x:x+w]

        try:
            face_resized = cv2.resize(face_crop, (224, 224)) / 255.0
        except Exception:
            continue

        face_input = np.expand_dims(face_resized, axis=0)

        prediction = model.predict(face_input, verbose=0)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        confidence = prediction[predicted_index]

        # Tampilkan hanya jika confidence tinggi (> 0.8 misalnya)
        if confidence > 0.8:
            # Convert crop image ke base64 untuk ditampilkan di web
            cropped_pil = Image.fromarray(face_crop)
            buffered = BytesIO()
            cropped_pil.save(buffered, format="JPEG")
            face_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            results.append({
                'label': predicted_label,
                'confidence': round(float(confidence), 2),
                'image': face_base64
            })

    return JsonResponse({'results': results})
