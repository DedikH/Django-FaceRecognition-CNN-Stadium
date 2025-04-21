from django.urls import path
from . import views

urlpatterns = [
    path('', views.scan_view, name='scan'),
    path('camera_feed/', views.camera_feed, name='camera_feed'),
    path('blacklist/<str:label>/', views.show_blacklist, name='blacklist_detail'),
]
