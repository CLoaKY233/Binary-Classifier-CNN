import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model #type:ignore
import cv2

def test_image(model_path, image_path):
    # Load the saved model
    model = load_model(model_path)

    # Preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to open.")

    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.reshape(img, (1, 256, 256, 1))

    # Make prediction
    prediction = model.predict(img)

    # Interpret prediction
    class_index = 1 if prediction[0][0] > 0.5 else 0
    class_name = "Cat" if class_index == 1 else "Dog"
    confidence = prediction[0][0] if class_index == 1 else 1 - prediction[0][0]

    # Display the image and prediction
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f"Prediction: {class_name} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2f}")

# Usage
model_path = "dogcatclassifier.h5"  # Path to your saved model
image_path = "kutta1.jpg"  # Path to the image you want to test
test_image(model_path, image_path)
