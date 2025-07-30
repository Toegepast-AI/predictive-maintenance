import argparse
import os
import numpy as np
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


def iou_score(pred, mask):
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    return intersection / union if union != 0 else 1.0

def dice_score(pred, mask):
    intersection = np.logical_and(pred, mask).sum()
    return 2 * intersection / (pred.sum() + mask.sum()) if (pred.sum() + mask.sum()) != 0 else 1.0

def pixel_accuracy(pred, mask):
    return (pred == mask).sum() / pred.size

def main():
    parser = argparse.ArgumentParser(description="Crack Segmentation Evaluation using Hugging Face")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with test images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory with ground truth masks')
    parser.add_argument('--model_name', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512', help='Hugging Face model name')
    args = parser.parse_args()

    feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name)

    crack_metrics = []
    noncrack_metrics = []
    all_metrics = []

    image_files = [f for f in os.listdir(args.image_dir) if f.endswith('.jpg')]
    for img_name in image_files:
        img_path = os.path.join(args.image_dir, img_name)
        mask_path = os.path.join(args.mask_dir, img_name)
        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_name}, skipping.")
            continue
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        # Binarize mask: cracks=1, background=0
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
        # Binarize prediction: cracks=1, background=0
        pred = (pred > 0).astype(np.uint8)

        iou = iou_score(pred, mask)
        dice = dice_score(pred, mask)
        acc = pixel_accuracy(pred, mask)
        metrics = (iou, dice, acc)
        all_metrics.append(metrics)

        if img_name.startswith('noncrack'):
            noncrack_metrics.append(metrics)
        else:
            crack_metrics.append(metrics)

        print(f"{img_name}: IoU={iou:.4f}, Dice={dice:.4f}, Acc={acc:.4f}")

    def avg_metric(metric_list):
        arr = np.array(metric_list)
        return arr.mean(axis=0) if len(arr) > 0 else (0, 0, 0)

    crack_avg = avg_metric(crack_metrics)
    noncrack_avg = avg_metric(noncrack_metrics)
    all_avg = avg_metric(all_metrics)

    print("\nAverage metrics:")
    print(f"Crack images:    IoU={crack_avg[0]:.4f}, Dice={crack_avg[1]:.4f}, Acc={crack_avg[2]:.4f}")
    print(f"Noncrack images: IoU={noncrack_avg[0]:.4f}, Dice={noncrack_avg[1]:.4f}, Acc={noncrack_avg[2]:.4f}")
    print(f"All images:      IoU={all_avg[0]:.4f}, Dice={all_avg[1]:.4f}, Acc={all_avg[2]:.4f}")

if __name__ == "__main__":
    main()
