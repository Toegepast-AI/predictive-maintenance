import pickle

def load_model(model_path):
    """Load trained model from file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_maintenance(model, features):
    """Predict maintenance need from features using the model."""
    prediction = model.predict(features)
    return prediction[0]
