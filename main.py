import argparse
from app.image_processing import preprocess_image
from app.feature_extraction import extract_features
from app.model import load_model, predict_maintenance


def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance Computer Vision App")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/model.pkl', help='Path to trained model')
    args = parser.parse_args()

    # Preprocess image
    image = preprocess_image(args.image)
    features = extract_features(image)
    model = load_model(args.model)
    prediction = predict_maintenance(model, features)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
