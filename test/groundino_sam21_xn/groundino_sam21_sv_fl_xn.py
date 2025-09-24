# app.py - Flask server for Grounding DINO and SAM2.1 APIs

import os
import cv2
import json
import torch
import numpy as np
import base64
from flask import Flask, request, jsonify
from pathlib import Path
from torchvision.ops import box_convert
import pycocotools.mask as mask_util

# Import Grounding DINO modules
from grounding_dino.groundingdino.util.inference import load_model, predict
from grounding_dino.groundingdino.util.inference import load_image as dino_load_image

# Import SAM2 modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Setup model paths based on environment
# BASE_DIR = os.environ.get("MODEL_BASE_DIR", "/home/admin/workspace/Grounded-SAM-2")
BASE_DIR = os.environ.get("MODEL_BASE_DIR", "/media/yons/WIN10/prog/seamless_communication/Grounded_SAM21")

# Grounding DINO config
# GROUNDING_DINO_CONFIG = os.path.join(BASE_DIR, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# GROUNDING_DINO_CHECKPOINT = os.path.join(BASE_DIR, "gdino_checkpoints/groundingdino_swint_ogc.pth")
GROUNDING_DINO_CONFIG = "/media/yons/WIN10/prog/seamless_communication/Grounded_SAM21/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/media/yons/WIN10/prog/seamless_communication/Grounded_SAM21/gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# SAM2 config
# SAM2_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints/sam2.1_hiera_large.pt")
# SAM2_MODEL_CONFIG = os.path.join(BASE_DIR, "configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Model initialization function
def initialize_models():
    print("Initializing models on device:", DEVICE)

    # Enable mixed precision for better performance on compatible GPUs
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize Grounding DINO
    print(f"Loading Grounding DINO from: {GROUNDING_DINO_CHECKPOINT}")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    # Initialize SAM2
    print(f"Loading SAM2 from: {SAM2_CHECKPOINT}")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return grounding_model, sam2_predictor

# Initialize models at startup
grounding_model, sam2_predictor = initialize_models()

def decode_image(encoded_image):
    """Decode base64 image to numpy array"""
    image_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def encode_mask(mask):
    """Encode mask as RLE format"""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

@app.route('/api/grounding_dino', methods=['POST'])
def grounding_dino_detect():
    """API endpoint for Grounding DINO object detection"""
    try:
        data = request.json

        # Get text prompt and image from request
        text_prompt = data.get('text_prompt', '')
        encoded_image = data.get('image', '')
        box_threshold = float(data.get('box_threshold', BOX_THRESHOLD))
        text_threshold = float(data.get('text_threshold', TEXT_THRESHOLD))

        # Validate inputs
        if not text_prompt or not encoded_image:
            return jsonify({"error": "Missing text_prompt or image"}), 400

        # Process text prompt - ensure it ends with periods
        if not all(term.strip().endswith('.') for term in text_prompt.split()):
            text_prompt = ' '.join([term.strip() + '.' if not term.strip().endswith('.') else term.strip()
                                    for term in text_prompt.split()])

        # Decode image（内存中解码，得到 BGR 格式的 numpy 数组）
        img = decode_image(encoded_image)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        image_source = img.copy()
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = image_rgb / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32).to(DEVICE)

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Convert boxes to absolute coordinates
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()

        # Prepare response
        results = {
            "boxes": boxes_xyxy,
            "confidences": confidences.numpy().tolist(),
            "labels": labels,
            "image_height": h,
            "image_width": w
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sam2', methods=['POST'])
def sam2_segment():
    """API endpoint for SAM2 segmentation (修复参数验证逻辑)"""
    try:
        data = request.json

        encoded_image = data.get('image', '')
        boxes = data.get('boxes', [])

        if not encoded_image or not boxes:
            return jsonify({"error": "Missing boxes or image"}), 400

        # 解码图片
        img = decode_image(encoded_image)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # 处理图片
        image_source = img.copy()
        sam2_predictor.set_image(image_source)

        # 运行SAM2分割
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(boxes),  # 直接使用传入的boxes
            multimask_output=False,
        )

        # 处理掩码并返回结果
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask_rles = [encode_mask(mask) for mask in masks]

        results = {
            "masks_rle": mask_rles,
            "scores": scores.tolist(),
            "image_height": image_source.shape[0],
            "image_width": image_source.shape[1]
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/combined', methods=['POST'])
def combined_detection_segmentation():
    """API endpoint that combines Grounding DINO and SAM2 in one call（无临时文件版）"""
    try:
        data = request.json

        # Get text prompt and image from request
        text_prompt = data.get('text_prompt', '')
        encoded_image = data.get('image', '')
        box_threshold = float(data.get('box_threshold', BOX_THRESHOLD))
        text_threshold = float(data.get('text_threshold', TEXT_THRESHOLD))

        # Validate inputs
        if not text_prompt or not encoded_image:
            return jsonify({"error": "Missing text_prompt or image"}), 400

        # Process text prompt - ensure it ends with periods
        if not all(term.strip().endswith('.') for term in text_prompt.split()):
            text_prompt = ' '.join([term.strip() + '.' if not term.strip().endswith('.') else term.strip()
                                    for term in text_prompt.split()])

        # Decode image（内存中解码，得到 BGR 格式的 numpy 数组）
        img = decode_image(encoded_image)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        image_source = img.copy()
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = image_rgb / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32).to(DEVICE)

        # Run Grounding DINO prediction
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Convert boxes to absolute coordinates
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Set image for SAM2 predictor
        sam2_predictor.set_image(image_source)

        # Run SAM2 prediction
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )

        # Process masks（处理维度并编码为 RLE）
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask_rles = [encode_mask(mask) for mask in masks]

        # Prepare response
        results = {
            "boxes": boxes_xyxy.tolist(),
            "confidences": confidences.numpy().tolist(),
            "labels": labels,
            "masks_rle": mask_rles,
            "segmentation_scores": scores.tolist(),
            "image_height": h,
            "image_width": w
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)