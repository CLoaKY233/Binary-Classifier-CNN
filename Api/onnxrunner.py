import onnxruntime as ort
import numpy as np
import cv2

def test_onnx_model(image_path):
    # Load the ONNX model
    onnx_path = "saves/dogcatclassifier.onnx"
    ort_session = ort.InferenceSession(onnx_path)

    # Preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to open.")

    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=0)  # Add channel dimension

    # Run the model
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)

    # Process the output
    prediction = ort_outs[0]
    class_index = 1 if prediction[0][0] > 0.5 else 0
    class_name = "Cat" if class_index == 1 else "Dog"
    confidence = prediction[0][0] if class_index == 1 else 1 - prediction[0][0]

    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2f}")
    return class_name, confidence
