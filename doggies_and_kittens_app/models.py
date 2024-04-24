from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    result = models.CharField(max_length=50)
