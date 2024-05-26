# urls.py
from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path('classify_image/', views.classify_image, name='classify_image'),
    path('', lambda request: render(request, 'doggies_and_kittens_app/upload_image.html'), name='upload_image'),
]
