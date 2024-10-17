import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import os
from typing import Tuple
from models.models import Model1, Model2, Model3

# Constants
IMG_SIZE = 256  # Size to which each image will be resized
BATCH_SIZE = 32  # Number of samples per batch
EPOCHS = 5  # Number of epochs to train the model
LEARNING_RATE = 0.0001  # Learning rate for the optimizer

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels from the specified directory.
    Args:
    path (str): Path to the dataset directory.
    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the images and labels.
    """
    x_dataset = []  # List to store image data
    y_dataset = []  # List to store labels

    # Loop through each class (cats and dogs)
    for class_name in ['cats', 'dogs']:
        class_path = os.path.join(path, class_name)  # Path to the class directory
        label = 1 if class_name == 'cats' else 0  # Label for the class

        # Loop through each image in the class directory
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)  # Path to the image

            try:
                # Read the image in grayscale
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Resize the image to the specified size
                img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                x_dataset.append(img_arr)  # Add the image to the dataset
                y_dataset.append(label)  # Add the label to the dataset
            except Exception as e:
                print(f"{img_path} was not added. Error: {e}")

    x = np.array(x_dataset)  # Convert the list of images to a numpy array
    y = np.array(y_dataset)  # Convert the list of labels to a numpy array

    # Shuffle the dataset
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    return x[indices], y[indices]  # Return the shuffled dataset

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> float: #type:ignore
    """
    Train the model for one epoch.

    Args:
    model (nn.Module): The model to train.
    train_loader (DataLoader): DataLoader for the training data.
    criterion (nn.Module): Loss function.
    optimizer (optim.Optimizer): Optimizer.
    device (torch.device): Device to run the training on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize the running loss

    # Loop through each batch in the training data
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels.unsqueeze(1).float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        running_loss += loss.item()  # Update the running loss

    return running_loss / len(train_loader)  # Return the average loss

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the model on the test data.

    Args:
    model (nn.Module): The model to evaluate.
    test_loader (DataLoader): DataLoader for the test data.
    device (torch.device): Device to run the evaluation on.

    Returns:
        float: The accuracy of the model on the test data.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Initialize the number of correct predictions
    total = 0  # Initialize the total number of samples

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Loop through each batch in the test data
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            outputs = model(inputs)  # Forward pass
            predicted = (outputs > 0.5).float()  # Convert outputs to binary predictions
            total += labels.size(0)  # Update the total number of samples
            correct += (predicted.squeeze() == labels).sum().item()  # Update the number of correct predictions

    return 100 * correct / total  # Return the accuracy

def main():
    """
    Main function to load data, train the model, and evaluate it.
    """
    train_path = r"dataset/training_set/training_set"  # Path to the training data
    test_path = r"dataset/test_set/test_set"  # Path to the test data

    # Load the training and test data
    x_train, y_train = load_data(train_path)
    x_test, y_test = load_data(test_path)

    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert to PyTorch tensors
    x_train = torch.FloatTensor(x_train).unsqueeze(1)  # Add channel dimension
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test).unsqueeze(1)  # Add channel dimension
    y_test = torch.FloatTensor(y_test)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ask user for model choice
    model_choice = input("Choose a model (1/2/3): ").strip()
    if model_choice == '1':
        model = Model1().to(device)
    elif model_choice == '2':
        model = Model2().to(device)
    elif model_choice == '3':
        model = Model3().to(device)
    else:
        print("Invalid choice. Defaulting to Model1.")
        model = Model1().to(device)

    # Initialize weights similar to TensorFlow's default
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #type:ignore

    # Train the model for the specified number of epochs
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Evaluate the final model on the test data
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    # Save the trained model
    savename = input("Enter the name of the model to save: ")
    torch.save(model.state_dict(), f"models/{savename}.pth")

if __name__ == "__main__":
    main()
