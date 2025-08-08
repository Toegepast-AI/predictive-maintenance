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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Please upload both image and mask files.'}), 400
        
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        # Validate file sizes to prevent memory issues
        if image_file.content_length and image_file.content_length > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({'error': 'Image file too large. Maximum 10MB allowed.'}), 400
        if mask_file.content_length and mask_file.content_length > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({'error': 'Mask file too large. Maximum 10MB allowed.'}), 400
        
        # Process files
        image = Image.open(image_file).convert("RGB")
        mask = Image.open(mask_file).convert("L")
        
        # Resize if too large to speed up processing
        max_size = (512, 512)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        if mask.size[0] > max_size[0] or mask.size[1] > max_size[1]:
            mask.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)

        # Model inference
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

        # Calculate metrics
        iou = iou_score(pred, mask)
        dice = dice_score(pred, mask)
        acc = pixel_accuracy(pred, mask)

        return jsonify({
            'IoU': float(iou),
            'Dice': float(dice),
            'Accuracy': float(acc),
            'image_size': image.size,
            'processing_status': 'success'
        })
    
    except Exception as e:
        # Log the error and return a proper error response
        print(f"Error processing prediction: {str(e)}")
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'processing_status': 'failed'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
