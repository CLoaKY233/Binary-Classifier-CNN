# Cats and Dogs Classifier

Welcome to the Cats and Dogs Classifier project! This repository contains a Convolutional Neural Network (CNN) implementation using both PyTorch and TensorFlow/Keras to classify images of cats and dogs. The project is designed to be easy to understand and use, even for those new to deep learning.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to classify images of cats and dogs using CNNs. We provide implementations in both PyTorch and TensorFlow/Keras, allowing you to choose your preferred framework. The models are trained on a dataset of cat and dog images, and the trained models can be used to predict the class of new images.

## Dataset

The dataset used for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data). It contains separate folders for training and testing images of cats and dogs.

## Project Structure

```
Cats and Dogs Classifier/
│
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   └── dogs/
│   └── test_set/
│       ├── cats/
│       └── dogs/
│
├── models/
│   └── models.py
│
├── Cats and Dogs classifier/
│   ├── PyTorchCNN.py
│   └── CNN_Binaryclassifier.py
│
├── README.md
└── requirements.txt
```

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/cats-and-dogs-classifier.git
    cd cats-and-dogs-classifier
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset:**

    Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data) and place it in the `dataset` directory as shown in the project structure.

## Usage

### PyTorch Implementation

1. **Navigate to the PyTorch script:**

    ```bash
    cd "Cats and Dogs classifier"
    ```

2. **Run the PyTorch script:**

    ```bash
    python PyTorchCNN.py
    ```

3. **Follow the prompts:**

    - Choose a model (1, 2, or 3).
    - Enter the name to save the trained model.

### TensorFlow/Keras Implementation

1. **Navigate to the TensorFlow/Keras script:**

    ```bash
    cd "Cats and Dogs classifier"
    ```

2. **Run the TensorFlow/Keras script:**

    ```bash
    python CNN_Binaryclassifier.py
    ```

3. **Follow the prompts:**

    - The script will automatically train the model and save it as `dogcatclassifier.h5`.

## Model Architectures

### PyTorch Models

- **Model1:** A simple CNN with three convolutional layers followed by max-pooling layers and three fully connected layers.
- **Model2:** Similar to Model1 but with batch normalization layers after each convolutional layer.
- **Model3:** Similar to Model2 but with increased capacity in the fully connected layers.

### TensorFlow/Keras Model

- A simple CNN with three convolutional layers followed by max-pooling layers and three fully connected layers.

## Training and Evaluation

### PyTorch

- The training and evaluation process is handled in the `PyTorchCNN.py` script.
- The script trains the model for a specified number of epochs and evaluates it on the test set.
- The final model is saved to the `models` directory.

### TensorFlow/Keras

- The training and evaluation process is handled in the `CNN_Binaryclassifier.py` script.
- The script trains the model for a specified number of epochs and evaluates it on the test set.
- The final model is saved as `dogcatclassifier.h5`.

## Results

- The training and validation accuracy/loss are plotted and displayed after training.
- The final test accuracy is printed in the console.

## Contributing

We welcome contributions to this project! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for checking out the Cats and Dogs Classifier project! We hope you find it useful and educational. If you have any questions or need further assistance, please feel free to reach out. Happy coding! 🐱🐶
