from django.urls import path
from . import views_video

urlpatterns = [
    path('', views_video.index, name='index'),
    path('video_simulation_feed', views_video.video_simulation_feed, name='video_simulation_feed'),
    path('snapshot_api_simulation', views_video.snapshot_api_simulation, name='snapshot_api_simulation'),
    path('detail/<str:nama>/', views_video.detail_pelanggar, name='detail'),

    path('pelaku_list/', views_video.pelaku_list, name='pelaku_list'),
    path('create/', views_video.pelaku_create, name='pelaku_create'),
    path('edit/<int:pk>/', views_video.pelaku_update, name='pelaku_update'),
    path('delete/<int:pk>/', views_video.pelaku_delete, name='pelaku_delete'),
    path('foto/<int:pk>/', views_video.pelaku_foto, name='pelaku_foto'),
    ] 