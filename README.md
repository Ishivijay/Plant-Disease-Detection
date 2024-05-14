# Plant Disease Detection using OpenCV

This repository contains Python code for detecting plant diseases using computer vision techniques with OpenCV.

## Overview

Plant diseases can significantly reduce crop yields and affect food security. Early detection and management of these diseases are essential to prevent crop losses. This project aims to develop a computer vision-based system for automatically detecting plant diseases from images of plant leaves.

## Features

- Loads images of healthy and diseased plant leaves from specified directories.
- Preprocesses images by resizing them to a fixed size and converting them to grayscale.
- Trains a simple K Nearest Neighbors (KNN) model for classification using OpenCV.
- Predicts the class (healthy or diseased) of a single plant leaf image.

## Requirements

- Python 3.x
- OpenCV (opencv-python)
- NumPy

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Ishivijay/plant-disease-detection.git
    cd plant-disease-detection
    ```

2. **Organize your dataset:**

    - Create two separate directories for healthy and diseased plant images.
    - Place the images of healthy plants in one directory and images of diseased plants in another directory.

3. **Update the paths in the code:**

    - Update the paths to the directories containing healthy and diseased plant images in the Python code for training of data.
    - Additionally, update the path to the test image in the Python code.

4. **Install the required packages:**

    ```bash
    pip install opencv-python numpy
    ```

5. **Run the code:**

    ```bash
    python plant_disease_detection.py
    ```

6. **View the prediction:**

    - The code will predict whether the provided test image contains a healthy or diseased plant.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request.
