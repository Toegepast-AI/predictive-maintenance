import cv2
import numpy as np

def preprocess_image(image_path):
    """Load and preprocess image for feature extraction."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Example: convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Example: resize
    resized = cv2.resize(gray, (224, 224))
    return resized
