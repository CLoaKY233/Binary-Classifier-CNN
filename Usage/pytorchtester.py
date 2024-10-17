import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.models import Model1

def test_image(model_path, image_path):
    # Load the saved model
    model = Model1()
    state_dic = torch.load(model_path)
    model.load_state_dict(state_dic)
    model.eval()  # Set the model to evaluation mode

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to open.")

    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0

    # Convert to PyTorch tensor and add batch and channel dimensions
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension #type:ignore

    # Make prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    # Interpret prediction
    probability = torch.sigmoid(prediction).item()
    class_index = 1 if probability > 0.5 else 0
    class_name = "Cat" if class_index == 1 else "Dog"
    confidence = probability if class_index == 1 else 1 - probability

    # Display the image and prediction
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f"Prediction: {class_name} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2f}")

# Usage
model_path = "saves/cat_dog_classifier.pth"  # Path to your saved PyTorch model
image_path = "images/cat4.webp"  # Path to the image you want to test
test_image(model_path, image_path)
