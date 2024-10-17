import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Constants
IMG_SIZE = 256
CLASSES = {0: "Dog", 1: "Cat"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CatDogDataset(Dataset):
    def __init__(self, path):
        self.data = []
        cats_path = os.path.join(path, "cats")
        dogs_path = os.path.join(path, "dogs")

        for img in os.listdir(cats_path):
            self.data.append((os.path.join(cats_path, img), 1))
        for img in os.listdir(dogs_path):
            self.data.append((os.path.join(dogs_path, img), 0))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += ((outputs > 0.5) == labels.unsqueeze(1)).float().sum()

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                val_correct += ((outputs > 0.5) == labels.unsqueeze(1)).float().sum()

        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies

# def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 2, 1)
#     plt.plot(train_accuracies, label='Training Accuracy')
#     plt.plot(val_accuracies, label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    image = transform(image)
    return image.unsqueeze(0)

def display_prediction(image_path, predicted_class, prediction):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_class}\nProbability: {prediction[0][0]:.4f}")
    plt.axis('off')
    plt.show()

def main():
    train_path = "dataset/training_set/training_set"
    test_path = "dataset/test_set/test_set"

    train_dataset = CatDogDataset(train_path)
    test_dataset = CatDogDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CatDogClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=5
    )

    # plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)

    torch.save(model.state_dict(), "saves/dogcatclassifier.pth")

    sample_image_path = "/content/drive/MyDrive/Colab Notebooks/images.jpeg"
    input_tensor = preprocess_image(sample_image_path).to(DEVICE)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
    predicted_class = CLASSES[round(prediction.item())]
    display_prediction(sample_image_path, predicted_class, prediction.cpu().numpy())

if __name__ == "__main__":
    main()
