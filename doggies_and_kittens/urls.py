from doggies_and_kittens_app import views
from django.urls import include, path
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('doggies_and_kittens_app.urls')),
]
