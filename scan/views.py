from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
from .face_recognition import predict_face

def gen_frames():
    cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            status = predict_face(frame)

            # Tampilkan status
            color = (0, 255, 0) if status == "Safe" else (0, 0, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def camera_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def scan_view(request):
    return render(request, 'scan/scan.html')
