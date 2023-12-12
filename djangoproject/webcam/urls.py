from django.urls import path
from . import views
urlpatterns = [
    path('webcam/', views.webcam, name='webcam'),
    path('index/', views.index, name='index'),
    path('addface/', views.addface, name='addface'),
    path('create/', views.create, name='create'),
    # path('createface/', views.create_face, name='createface'),
]
