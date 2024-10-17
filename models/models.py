# models/models.py
import torch.nn as nn
import torch
import torch.nn.functional as F

class Model1(nn.Module):
    """
    Convolutional Neural Network Model1 for binary classification.

    This model consists of three convolutional layers followed by max-pooling layers,
    and three fully connected layers. Dropout is applied after the first fully connected layer
    to prevent overfitting. The final output layer uses a sigmoid activation function to produce
    a binary classification output.

    Attributes:
    conv1 (nn.Conv2d): First convolutional layer.
    conv2 (nn.Conv2d): Second convolutional layer.
    conv3 (nn.Conv2d): Third convolutional layer.
    pool (nn.MaxPool2d): Max pooling layer.
    flatten (nn.Flatten): Flatten layer to convert 2D feature maps to 1D feature vectors.
    fc1 (nn.Linear): First fully connected layer.
    fc2 (nn.Linear): Second fully connected layer.
    fc3 (nn.Linear): Third fully connected layer (output layer).
    dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self):
        super(Model1, self).__init__()
        # Define the first convolutional layer with 64 filters, kernel size 3x3, and same padding
        self.conv1 = nn.Conv2d(1, 64, 3, padding='same')
        # Define the second convolutional layer with 128 filters, kernel size 3x3, and same padding
        self.conv2 = nn.Conv2d(64, 128, 3, padding='same')
        # Define the third convolutional layer with 256 filters, kernel size 3x3, and same padding
        self.conv3 = nn.Conv2d(128, 256, 3, padding='same')
        # Define the max pooling layer with kernel size 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # Define the flatten layer to convert 2D feature maps to 1D feature vectors
        self.flatten = nn.Flatten()
        # Define the first fully connected layer with 128 output features
        self.fc1 = nn.Linear(256 * 32 * 32, 128)
        # Define the second fully connected layer with 64 output features
        self.fc2 = nn.Linear(128, 64)
        # Define the third fully connected layer with 1 output feature (binary classification)
        self.fc3 = nn.Linear(64, 1)
        # Define the dropout layer with a dropout probability of 0.5
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 1, 64, 64).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1) with sigmoid activation.
        """
        # Apply the first convolutional layer, followed by ReLU activation and max pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply the second convolutional layer, followed by ReLU activation and max pooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Apply the third convolutional layer, followed by ReLU activation and max pooling
        x = self.pool(torch.relu(self.conv3(x)))
        # Flatten the feature maps to a 1D feature vector
        x = self.flatten(x)
        # Apply the first fully connected layer, followed by ReLU activation and dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # Apply the second fully connected layer, followed by ReLU activation
        x = torch.relu(self.fc2(x))
        # Apply the third fully connected layer, followed by sigmoid activation for binary classification
        x = torch.sigmoid(self.fc3(x))
        return x


class Model2(nn.Module):
    """
    Convolutional Neural Network Model2 for binary classification with Batch Normalization.

    This model consists of three convolutional layers with batch normalization, followed by max-pooling layers,
    and three fully connected layers. Dropout is applied after the first fully connected layer
    to prevent overfitting. The final output layer uses a sigmoid activation function to produce
    a binary classification output.

    Attributes:
    conv1 (nn.Conv2d): First convolutional layer.
    bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
    conv2 (nn.Conv2d): Second convolutional layer.
    bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolutional layer.
    conv3 (nn.Conv2d): Third convolutional layer.
    bn3 (nn.BatchNorm2d): Batch normalization layer after the third convolutional layer.
    pool (nn.MaxPool2d): Max pooling layer.
    flatten (nn.Flatten): Flatten layer to convert 2D feature maps to 1D feature vectors.
    fc1 (nn.Linear): First fully connected layer.
    fc2 (nn.Linear): Second fully connected layer.
    fc3 (nn.Linear): Third fully connected layer (output layer).
    dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self):
        super(Model2, self).__init__()
        # Define the first convolutional layer with 64 filters, kernel size 3x3, and same padding
        self.conv1 = nn.Conv2d(1, 64, 3, padding='same')
        # Define the batch normalization layer after the first convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        # Define the second convolutional layer with 128 filters, kernel size 3x3, and same padding
        self.conv2 = nn.Conv2d(64, 128, 3, padding='same')
        # Define the batch normalization layer after the second convolutional layer
        self.bn2 = nn.BatchNorm2d(128)
        # Define the third convolutional layer with 256 filters, kernel size 3x3, and same padding
        self.conv3 = nn.Conv2d(128, 256, 3, padding='same')
        # Define the batch normalization layer after the third convolutional layer
        self.bn3 = nn.BatchNorm2d(256)
        # Define the max pooling layer with kernel size 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # Define the flatten layer to convert 2D feature maps to 1D feature vectors
        self.flatten = nn.Flatten()
        # Define the first fully connected layer with 128 output features
        self.fc1 = nn.Linear(256 * 32 * 32, 128)
        # Define the second fully connected layer with 64 output features
        self.fc2 = nn.Linear(128, 64)
        # Define the third fully connected layer with 1 output feature (binary classification)
        self.fc3 = nn.Linear(64, 1)
        # Define the dropout layer with a dropout probability of 0.5
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 1, 64, 64).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1) with sigmoid activation.
        """
        # Apply the first convolutional layer, followed by batch normalization, ReLU activation, and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Apply the second convolutional layer, followed by batch normalization, ReLU activation, and max pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Apply the third convolutional layer, followed by batch normalization, ReLU activation, and max pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten the feature maps to a 1D feature vector
        x = self.flatten(x)
        # Apply the first fully connected layer, followed by ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Apply the second fully connected layer, followed by ReLU activation
        x = F.relu(self.fc2(x))
        # Apply the third fully connected layer, followed by sigmoid activation for binary classification
        x = torch.sigmoid(self.fc3(x))
        return x


class Model3(nn.Module):
    """
    Convolutional Neural Network Model3 for binary classification with Batch Normalization and increased capacity.

    This model consists of three convolutional layers with batch normalization, followed by max-pooling layers,
    and three fully connected layers with increased capacity. Dropout is applied after the first fully connected layer
    to prevent overfitting. The final output layer uses a sigmoid activation function to produce
    a binary classification output.

    Attributes:
    conv1 (nn.Conv2d): First convolutional layer.
    bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
    conv2 (nn.Conv2d): Second convolutional layer.
    bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolutional layer.
    conv3 (nn.Conv2d): Third convolutional layer.
    bn3 (nn.BatchNorm2d): Batch normalization layer after the third convolutional layer.
    pool (nn.MaxPool2d): Max pooling layer.
    flatten (nn.Flatten): Flatten layer to convert 2D feature maps to 1D feature vectors.
    fc1 (nn.Linear): First fully connected layer with increased capacity.
    fc2 (nn.Linear): Second fully connected layer with increased capacity.
    fc3 (nn.Linear): Third fully connected layer (output layer).
    dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self):
        super(Model3, self).__init__()
        # Define the first convolutional layer with 64 filters, kernel size 3x3, and same padding
        self.conv1 = nn.Conv2d(1, 64, 3, padding='same')
        # Define the batch normalization layer after the first convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        # Define the second convolutional layer with 128 filters, kernel size 3x3, and same padding
        self.conv2 = nn.Conv2d(64, 128, 3, padding='same')
        # Define the batch normalization layer after the second convolutional layer
        self.bn2 = nn.BatchNorm2d(128)
        # Define the third convolutional layer with 256 filters, kernel size 3x3, and same padding
        self.conv3 = nn.Conv2d(128, 256, 3, padding='same')
        # Define the batch normalization layer after the third convolutional layer
        self.bn3 = nn.BatchNorm2d(256)
        # Define the max pooling layer with kernel size 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # Define the flatten layer to convert 2D feature maps to 1D feature vectors
        self.flatten = nn.Flatten()
        # Define the first fully connected layer with 512 output features (increased capacity)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        # Define the second fully connected layer with 128 output features (increased capacity)
        self.fc2 = nn.Linear(512, 128)
        # Define the third fully connected layer with 1 output feature (binary classification)
        self.fc3 = nn.Linear(128, 1)
        # Define the dropout layer with a dropout probability of 0.5
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 1, 64, 64).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1) with sigmoid activation.
        """
        # Apply the first convolutional layer, followed by batch normalization, ReLU activation, and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Apply the second convolutional layer, followed by batch normalization, ReLU activation, and max pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Apply the third convolutional layer, followed by batch normalization, ReLU activation, and max pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten the feature maps to a 1D feature vector
        x = self.flatten(x)
        # Apply the first fully connected layer with increased capacity, followed by ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Apply the second fully connected layer with increased capacity, followed by ReLU activation
        x = F.relu(self.fc2(x))
        # Apply the third fully connected layer, followed by sigmoid activation for binary classification
        x = torch.sigmoid(self.fc3(x))
        return x
