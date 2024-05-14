import cv2
import os
import numpy as np

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Load healthy and diseased plant images
healthy_images = load_images_from_folder("Plant_Leaf_Diseases_dataset\With_augmentation\Apple___healthy")
diseased_images = load_images_from_folder("Plant_Leaf_Diseases_dataset\With_augmentation\Apple___Black_rot")

# Preprocess images (resize, convert to grayscale)
def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (128, 128))  # Resize images to a fixed size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        processed_images.append(gray)
    return processed_images

healthy_images_processed = preprocess_images(healthy_images)
diseased_images_processed = preprocess_images(diseased_images)

# Convert images to the correct data type (float32) and reshape them
X = np.array(healthy_images_processed + diseased_images_processed, dtype=np.float32)
X = X.reshape(X.shape[0], -1)  # Flatten images
y = np.array([0] * len(healthy_images_processed) + [1] * len(diseased_images_processed))  # 0 for healthy, 1 for diseased

# Train the model
model = cv2.ml.KNearest_create()
model.train(X, cv2.ml.ROW_SAMPLE, y)

# Function to predict class of a single image
def predict_image(img):
    gray = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.reshape(1, -1)  # Flatten image
    _, result, _, _ = model.findNearest(np.array(gray, dtype=np.float32), k=1)
    return result.ravel()[0]

# Test the model on new images
test_image = cv2.imread("Plant_Leaf_Diseases_dataset\With_augmentation\Apple___Black_rot\image (2).JPG")
prediction = predict_image(test_image)
if prediction == 0:
    print("Healthy plant")
else:
    print("Diseased plant")
