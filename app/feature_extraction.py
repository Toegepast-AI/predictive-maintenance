import numpy as np

def extract_features(image):
    """Extract features from preprocessed image."""
    # Example: flatten image as features
    features = image.flatten()
    return features.reshape(1, -1)
