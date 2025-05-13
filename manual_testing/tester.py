import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image
import os
import uuid
import time

# Load model CNN
model = tf.keras.models.load_model(
    'Z:/All Portfolio and Project/SEMOGA JADI/Skema Pak Muklis/paMuklisFace/face_recognition_mobilenetv2.h5'
)

# Daftar label
labels = [
    'Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan',
    'Andy Samberg', 'Anushka Sharma', 'Billie Eilish', 'Brad Pitt',
    'Camila Cabello', 'Charlize Theron', 'Courtney Cox', 'Dwayne Johnson',
    'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan',
    'Hugh Jackman', 'Jessica Alba'
]

# Folder input & output
input_path = 'Z:/All Portfolio and Project/SEMOGA JADI/Skema Pak Muklis/paMuklisFace/testings.jpg'
save_folder = 'saved_faces'
os.makedirs(save_folder, exist_ok=True)

# Load gambar
image = cv2.imread(input_path)
if image is None:
    print("Gambar tidak ditemukan:", input_path)
    exit()

# Deteksi wajah
detector = MTCNN()
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(rgb)

if not faces:
    print("Tidak ada wajah terdeteksi.")
else:
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb[y:y+h, x:x+w]

        try:
            # Preprocessing untuk CNN
            face_resized = cv2.resize(face_crop, (224, 224)) / 255.0
            face_input = np.expand_dims(face_resized, axis=0)

            # Prediksi
            prediction = model.predict(face_input, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]
            confidence = prediction[predicted_index]

            print(f"Wajah ke-{i+1}: {predicted_label} ({confidence*100:.2f}%)")

            # Simpan crop wajah
            filename = f"{predicted_label}_{int(time.time())}_{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(save_folder, filename)
            cv2.imwrite(save_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            print("Disimpan ke:", save_path)

            # Tampilkan hasil
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f'{predicted_label} ({confidence*100:.1f}%)', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print("Gagal memproses wajah:", e)

    cv2.imshow("Hasil Deteksi", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
