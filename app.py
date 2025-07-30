from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

app = Flask(__name__)

MODEL_NAME = os.environ.get('MODEL_NAME', 'nvidia/segformer-b0-finetuned-ade-512-512')
feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

def iou_score(pred, mask):
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    return intersection / union if union != 0 else 1.0

def dice_score(pred, mask):
    intersection = np.logical_and(pred, mask).sum()
    return 2 * intersection / (pred.sum() + mask.sum()) if (pred.sum() + mask.sum()) != 0 else 1.0

def pixel_accuracy(pred, mask):
    return (pred == mask).sum() / pred.size

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({'error': 'Please upload both image and mask files.'}), 400
    image_file = request.files['image']
    mask_file = request.files['mask']
    image = Image.open(image_file).convert("RGB")
    mask = Image.open(mask_file).convert("L")
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )
    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    pred = (pred > 0).astype(np.uint8)

    iou = iou_score(pred, mask)
    dice = dice_score(pred, mask)
    acc = pixel_accuracy(pred, mask)

    return jsonify({
        'IoU': float(iou),
        'Dice': float(dice),
        'Accuracy': float(acc)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
