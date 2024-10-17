import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras #type:ignore
from tensorflow.keras.models import Sequential, load_model #type:ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten #type:ignore
from tensorflow.keras.optimizers import Adam #type:ignore

# Constants
IMG_SIZE = 256
CLASSES = {0: "Dog", 1: "Cat"}

def load_data(path):
    """
    Load and preprocess image data from the given path.

    Args:
        path (str): Path to the dataset directory.

    Returns:
        tuple: Preprocessed images and labels as numpy arrays.
    """
    x_dataset = []
    y_dataset = []

    # Load cat images
    cats_path = os.path.join(path, "cats")
    for img in os.listdir(cats_path):
        process_image(os.path.join(cats_path, img), x_dataset, y_dataset, 1)

    # Load dog images
    dogs_path = os.path.join(path, "dogs")
    for img in os.listdir(dogs_path):
        process_image(os.path.join(dogs_path, img), x_dataset, y_dataset, 0)

    # Convert to numpy arrays and shuffle
    x = np.array(x_dataset)
    y = np.array(y_dataset)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    return x[indices], y[indices]

def process_image(img_path, x_dataset, y_dataset, label):
    """
    Process a single image and add it to the dataset.

    Args:
        img_path (str): Path to the image file.
        x_dataset (list): List to store processed images.
        y_dataset (list): List to store labels.
        label (int): Label for the image (0 for dog, 1 for cat).
    """
    try:
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        x_dataset.append(img_arr)
        y_dataset.append(label)
    except Exception as e:
        print(f"{img_path} was not added. Error: {e}")

def create_model():
    """
    Create and compile the CNN model.

    Returns:
        keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def plot_training_history(history):
    """
    Plot the training and validation accuracy/loss.

    Args:
        history (keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype('float32') / 255.0
    return np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 1))

def display_prediction(image_path, predicted_class, prediction):
    """
    Display the original image with prediction results.

    Args:
        image_path (str): Path to the image file.
        predicted_class (str): Predicted class (Dog or Cat).
        prediction (float): Prediction probability.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_class}\nProbability: {prediction[0][0]:.4f}")
    plt.axis('off')
    plt.show()

def main():
    # Load and preprocess data
    train_path = "dataset/training_set/training_set"
    test_path = "dataset/test_set/test_set"

    x_train, y_train = load_data(train_path)
    x_test, y_test = load_data(test_path)

    # Reshape and normalize data
    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Create and train model
    model = create_model()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy}")

    # Plot training history
    plot_training_history(history)

    # Save model
    model.save("dogcatclassifier.h5")

    # Make a prediction on a sample image
    sample_image_path = "/content/drive/MyDrive/Colab Notebooks/images.jpeg"
    prediction = model.predict(preprocess_image(sample_image_path))
    predicted_class = CLASSES[round(prediction[0][0])]
    display_prediction(sample_image_path, predicted_class, prediction)

if __name__ == "__main__":
    main()
