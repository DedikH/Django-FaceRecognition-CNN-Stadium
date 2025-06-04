from django.urls import path
# from . import views_video
from . import views

urlpatterns = [
# Path: Views_video url start
#     path('', views_video.index, name='index'),
#     path('video_simulation_feed', views_video.video_simulation_feed, name='video_simulation_feed'),
#     path('snapshot_api_simulation', views_video.snapshot_api_simulation, name='snapshot_api_simulation'),
#     path('detail/<str:nama>/', views_video.detail_pelanggar, name='detail'),
#     path('scan', views_video.scan, name='scan'),
#     path('pelaku_list/', views_video.pelaku_list, name='pelaku_list'),
#     path('create/', views_video.pelaku_create, name='pelaku_create'),
#     path('edit/<int:pk>/', views_video.pelaku_update, name='pelaku_update'),
#     path('delete/<int:pk>/', views_video.pelaku_delete, name='pelaku_delete'),
#     path('foto/<int:pk>/', views_video.pelaku_foto, name='pelaku_foto'),
# path: Views_video url end

# Path: Views url start
    path('', views.index, name='index'),
    path('scan', views.scan, name='scan'),
    path('video_feed',views.video_feed, name='video_feed'),
    path('snapshot_api', views.snapshot_api, name='snapshot_api'),
    # CRUD Pelaku
    path('pelaku_list/', views.pelaku_list, name='pelaku_list'),
    path('create/', views.pelaku_create, name='pelaku_create'),
    path('edit/<int:pk>/', views.pelaku_update, name='pelaku_update'),
    path('delete/<int:pk>/',views.pelaku_delete, name='pelaku_delete'),
    path('foto/<int:pk>/', views.pelaku_foto, name='pelaku_foto'),
    path('detail/<str:nama>/',views.detail_pelanggar,name='detail'),
    # path: Views url end
    ] 