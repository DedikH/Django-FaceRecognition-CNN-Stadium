import base64
import cv2
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render
from .face_recognition import predict_face
from .get_pelanggar_labeling import get_pelanggar_by_label


# Fungsi untuk streaming video dan deteksi wajah
def gen_frames():
    cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            status, label = predict_face(frame)

            # Tambahkan status ke frame
            color = (0, 255, 0) if status == "Safe" else (0, 0, 255)
            cv2.putText(frame, f"{status}: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Encode frame ke format JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Kirim frame ke browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Endpoint untuk streaming kamera
def camera_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


# View utama scan
def scan_view(request):
    return render(request, 'scan/scan.html')


# Mengambil hasil prediksi wajah (status + data pelanggar jika ada)
def camera_result(request):
    cap = cv2.VideoCapture(1)
    success, frame = cap.read()
    cap.release()

    if not success:
        return JsonResponse({'status': 'Error', 'message': 'Kamera gagal dibuka'})

    status, label = predict_face(frame)

    if status == "Blacklist":
        data_pelanggar = get_pelanggar_by_label(label)
        if data_pelanggar:
            foto_base64 = base64.b64encode(data_pelanggar['foto']).decode('utf-8')
            return JsonResponse({
                'status': 'Blacklist',
                'nama': data_pelanggar['nama'],
                'umur': data_pelanggar['umur'],
                'kasus': data_pelanggar['kasus'],
                'tim': data_pelanggar['tim'],
                'foto': foto_base64
            })
    
    return JsonResponse({'status': 'Safe'})


# Halaman untuk menampilkan detail pelanggar
def show_blacklist(request, label):
    data = get_pelanggar_by_label(label)
    if data:
        foto_blob = data['foto']
        foto_base64 = base64.b64encode(foto_blob).decode('utf-8')

        return render(request, 'blacklist_detail.html', {
            'nama': data['nama'],
            'umur': data['umur'],
            'kasus': data['kasus'],
            'tim': data['tim'],
            'foto_base64': foto_base64
        })
    else:
        return HttpResponse("Pelanggar tidak ditemukan", status=404)
