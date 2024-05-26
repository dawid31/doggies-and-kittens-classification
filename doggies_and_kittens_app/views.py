# views.py
import torch
import torch.nn as nn
from torchvision import transforms
from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
import os
from django.conf import settings

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2))
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2))
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2))
        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*3*3, out_features=2))
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_4(x)
        x = self.classifier(x)
        return x

def load_model():
    model = ConvolutionalNeuralNetwork()
    model_path = os.path.join(settings.BASE_DIR, 'doggies_and_kittens_app', 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

model = load_model()

# Define the transformation for the input image
IMAGE_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        image_path = default_storage.save('tmp/' + image_file.name, image_file)
        
        # Open the image file
        image = Image.open(image_path)
        
        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Transform and add batch dimension

        # Move the tensor to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device)
        model.to(device)

        # Perform the classification
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Convert prediction to human-readable label
        label = 'Dog' if predicted.item() == 1 else 'Cat'

        # Clean up the saved image
        os.remove(image_path)
        
        return render(request, 'doggies_and_kittens_app/result.html', {'prediction': label})

    return render(request, 'doggies_and_kittens_app/upload_image.html')
