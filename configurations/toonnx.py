import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Trainers.TorchTrainer import CatDogClassifier

def convert_to_onnx(model_path, onnx_path):
    # Load the saved model
    model = CatDogClassifier()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    # Create dummy input with the same shape as your model expects
    dummy_input = torch.randn(1, 1, 256, 256)  # Adjust the shape as needed

    # Export the model to ONNX format
    torch.onnx.export(
        model,                      # Model to be exported
        dummy_input,                # Dummy input tensor
        onnx_path,                  # Output file name
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=11,           # ONNX version to export the model to
        do_constant_folding=True,   # Whether to execute constant folding for optimization
        input_names=['input'],      # Input tensor names
        output_names=['output'],    # Output tensor names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes
    )

    print(f"Model has been converted to ONNX format and saved as {onnx_path}")

if __name__ == "__main__":
    model_path = "saves/dogcatclassifier.pth"  # Path to your saved PyTorch model
    onnx_path = "saves/dogcatclassifier.onnx"  # Path to save the ONNX model

    convert_to_onnx(model_path, onnx_path)
