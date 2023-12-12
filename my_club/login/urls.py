from django.urls import path
from . import views

urlpatterns = [
    # Các URL patterns khác
    path('login/', views.login_view, name='login'),
    path('display/', views.display, name='display'),
]