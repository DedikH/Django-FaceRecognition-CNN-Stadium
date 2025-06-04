import cv2, time, uuid, os, base64, threading
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from io import BytesIO
from PIL import Image
from django.shortcuts import render, get_object_or_404, redirect
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Pelaku
from .forms import PelakuForm

# ─── 1) LOAD MODEL & LABEL ──────────────────────────────────────────────
model = tf.keras.models.load_model('model-cnn-facerecognition.h5')
labels = [
    'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Dedik Hasanah Wijaya', 'Dwayne Johnson', 'Roger Federer', 'Tom Cruise'
]  # daftar label

# ─── 2) INIT DETECTOR & STORAGE ────────────────────────────────────────
detector     = MTCNN()
SAVE_FOLDER  = "saved_faces"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ─── 3) BUAT 1 INSTANCE CAMERA & BUFFER ────────────────────────────────
cap = cv2.VideoCapture(1)
latest_frame = None
lock = threading.Lock()

def _update_frame_buffer():
    global latest_frame
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue
        with lock:
            latest_frame = frame.copy()
        # sedikit sleep supaya thread nggak full-cpu
        time.sleep(0.01)

# Jalankan background thread _sekali_ saat module di-import
threading.Thread(target=_update_frame_buffer, daemon=True).start()

# ─── 4) VIEWS BIASA ─────────────────────────────────────────────────────
def index(request):
    return render(request, 'detector/index.html')

def scan(request):
    return render(request, 'detector/scan.html')

def detail_pelanggar(request, nama):
    pelaku = get_object_or_404(Pelaku, nama=nama)
    return render(request, 'detector/detail_pelanggar.html', {'pelaku': pelaku})

# ─── 5) STREAMING VIEW ─────────────────────────────────────────────────
def gen_frames():
    """Yield JPEG frames dari latest_frame buffer."""
    global latest_frame
    while True:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

# ─── 6) SNAPSHOT API ───────────────────────────────────────────────────
@csrf_exempt
def snapshot_api(request):
    global latest_frame
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        return JsonResponse({'results': []})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    results = []

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        crop = rgb[y:y+h, x:x+w]

        # Langsung simpan ke saved_faces dulu
        fn = f"detected_{int(time.time())}_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(SAVE_FOLDER, fn)
        cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # ──> Load kembali gambar dari saved_faces untuk inference CNN
        try:
            img_loaded = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
            if img_loaded is None :
                print(f"[ERROR] Gambar tidak sesuai ukuran: {save_path}")
                continue
            
            img_resized = cv2.resize(img_loaded, (50, 50))
            normalized = img_resized / 255.0
            face_input = normalized.reshape(1, 50, 50, 1).astype('float32')

            prediction = model.predict(face_input, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]
            confidence = float(prediction[predicted_index])

        except Exception as e:
            print("Prediction error:", e)
            predicted_label = "Tidak dikenali"
            confidence = 0.0

        # Kirim hasil ke frontend
        pil = Image.fromarray(crop)
        buf = BytesIO(); pil.save(buf, 'JPEG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        results.append({
            'name': predicted_label,
            'confidence': round(confidence * 100, 2),
            'image': img_b64
        })

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
        form = PelakuForm(request.POST, request.FILES, instance=data)
        if form.is_valid():
            data.nama = form.cleaned_data['nama']
            data.umur = form.cleaned_data['umur']
            data.kasus = form.cleaned_data['kasus']
            data.tim_dukung = form.cleaned_data['tim_dukung']

            if request.FILES.get('foto_pelaku'):
                data.foto_pelaku = request.FILES['foto_pelaku']  # ✅ perbaikan

            data.save()
            return redirect('pelaku_list')
    else:
        form = PelakuForm(instance=data)

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