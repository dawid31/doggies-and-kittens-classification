from django.shortcuts import render, redirect
from .forms import UploadImageForm
from .models import UploadedImage
from .classifier import classify_image  

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            result = classify_image(uploaded_image.image.path) #Calling the classifier function
            uploaded_image.result = result
            uploaded_image.save()
            return redirect('result', pk=uploaded_image.pk)
    else:
        form = UploadImageForm()
    return render(request, 'doggies_and_kittens_app/upload_image.html', {'form': form})

def result(request, pk):
    uploaded_image = UploadedImage.objects.get(pk=pk)
    return render(request, 'doggies_and_kittens_app/result.html', {'uploaded_image': uploaded_image})
