import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from mtcnn import MTCNN
import base64
from io import BytesIO
from PIL import Image
import os
import uuid
import time
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404
from .models import Pelaku
from .forms import PelakuForm
from django.shortcuts import redirect, render
from django.http import HttpResponse

# Load model CNN
model = tf.keras.models.load_model('face_recognition_mobilenetv2.h5')
labels = [
    'Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan',
    'Andy Samberg', 'Anushka Sharma', 'Billie Eilish', 'Brad Pitt',
    'Camila Cabello', 'Charlize Theron', 'Courtney Cox', 'Dwayne Johnson',
    'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan',
    'Hugh Jackman', 'Jessica Alba'
]

# Path video
video_path = "Z:\\All Portfolio and Project\\SEMOGA JADI\\Skema Pak Muklis\\paMuklisFace\\videoplayback.mp4"

# Folder penyimpanan wajah
SAVE_FOLDER = "saved_faces"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Inisialisasi detektor wajah
detector = MTCNN()

# Inisialisasi views start
def index(request):
    return render(request, 'detector/index.html')

def detail_pelanggar(request, nama):
    data_pelaku = get_object_or_404(Pelaku, nama=nama)

    return render(request, 'detector/detail_pelanggar.html', {'pelaku': data_pelaku})
# Inisialisasi views end

# pemrosesan video start
def generate_video_stream():
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        cv2.waitKey(delay)
# pemrosesan video end


def video_simulation_feed(request):
    return StreamingHttpResponse(generate_video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def snapshot_api_simulation(request):
    cap = cv2.VideoCapture(video_path)
    results = []

    success, frame = cap.read()
    if not success:
        cap.release()
        return JsonResponse({'results': []})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb_frame[y:y+height, x:x+width]

        # ✅ Simpan setiap wajah yang terdeteksi
        filename_detected = f"detected_{int(time.time())}_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(SAVE_FOLDER, filename_detected)
        cv2.imwrite(save_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

        try:
            # ✅ Prediksi dengan CNN
            face_resized = cv2.resize(face_crop, (224, 224))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            prediction = model.predict(face_input, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]
            confidence = float(prediction[predicted_index])

            if confidence >= 0.25:
                # ✅ Konversi ke base64 untuk web
                pil_img = Image.fromarray(face_crop)
                buffer = BytesIO()
                pil_img.save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue()).decode()

                results.append({
                    'name': predicted_label,
                    'confidence': round(confidence * 100, 2),
                    'image': img_str
                })

        except Exception as e:
            print("Error processing face:", e)
            continue

    cap.release()
    return JsonResponse({'results': results})

# CRUD Pelaku Start
# List semua pelaku
def pelaku_list(request):
    data = Pelaku.objects.all()
    return render(request, 'detector/pelaku_list.html', {'data': data})

# Tambah pelaku old
def pelaku_create(request):
    if request.method == 'POST':
        form = PelakuForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('pelaku_list')
    else:
        form = PelakuForm()
    return render(request, 'detector/pelaku_form.html', {'form': form})
# Tambah pelaku old end


# Edit pelaku
def pelaku_update(request, pk):
    data = get_object_or_404(Pelaku, pk=pk)
    if request.method == 'POST':
        form = PelakuForm(request.POST, request.FILES, instance=data)
        if form.is_valid():
            form.save()
            return redirect('pelaku_list')
    else:
        form = PelakuForm(instance=data)
    return render(request, 'detector/pelaku_form.html', {'form': form})
# edit pelaku end

# Pelaku foto start
def pelaku_foto(request, pk):
    pelaku = get_object_or_404(Pelaku, pk=pk)
    if pelaku.foto_pelaku:
        return HttpResponse(pelaku.foto_pelaku, content_type='image/jpeg')
    return HttpResponse(status=404)
# pelaku foto end

# Edit Pelaku new
def pelaku_update(request, pk):
    data = get_object_or_404(Pelaku, pk=pk)

    if request.method == 'POST':
        form = PelakuForm(request.POST, request.FILES)
        if form.is_valid():
            data.nama = form.cleaned_data['nama']
            data.umur = form.cleaned_data['umur']
            data.kasus = form.cleaned_data['kasus']
            data.tim_dukung = form.cleaned_data['tim_dukung']
            if request.FILES.get('foto_pelaku'):
                data.foto_pelaku = request.FILES['foto_pelaku'].read()
            data.save()
            return redirect('pelaku_list')
    else:
        # isi awal form
        form = PelakuForm(initial={
            'nama': data.nama,
            'umur': data.umur,
            'kasus': data.kasus,
            'tim_dukung': data.tim_dukung
        })

    return render(request, 'detector/pelaku_form.html', {'form': form})
# Edit pelaku new end

# Hapus pelaku
def pelaku_delete(request, pk):
    data = get_object_or_404(Pelaku, pk=pk)
    if request.method == 'POST':
        data.delete()
        return redirect('pelaku_list')
    return render(request, 'detector/pelaku_confirm_delete.html', {'data': data})
# CRUD Pelaku End