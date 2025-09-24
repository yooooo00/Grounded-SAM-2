# client.py - Client for Grounding DINO and SAM2.1 APIs

import os
import cv2
import json
import base64
import numpy as np
import requests
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import supervision as sv
from PIL import Image

class GroundedSAM2Client:
    def __init__(self, base_url="http://localhost:5000"):
        """
        Initialize the client for Grounding DINO and SAM2.1 APIs

        Args:
            base_url: Base URL for the Flask API server
        """
        self.base_url = base_url
        self.grounding_dino_url = f"{base_url}/api/grounding_dino"
        self.sam2_url = f"{base_url}/api/sam2"
        self.combined_url = f"{base_url}/api/combined"

    def _encode_image(self, image_path):
        """
        Encode image as base64 string

        Args:
            image_path: Path to the image file

        Returns:
            base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _encode_image_from_array(self, image_array):
        """
        Encode image array as base64 string

        Args:
            image_array: Numpy array containing the image

        Returns:
            base64 encoded image string
        """
        success, encoded_image = cv2.imencode('.jpg', image_array)
        if not success:
            raise ValueError("Could not encode image")
        return base64.b64encode(encoded_image).decode('utf-8')

    def detect_objects(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Detect objects using Grounding DINO

        Args:
            image_path: Path to the image file
            text_prompt: Text prompt for object detection (e.g., "car. person.")
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text

        Returns:
            Detection results from Grounding DINO
        """
        # Ensure text prompt ends with periods
        if not all(term.strip().endswith('.') for term in text_prompt.split()):
            text_prompt = ' '.join([term.strip() + '.' if not term.strip().endswith('.') else term.strip()
                                    for term in text_prompt.split()])

        # Prepare request data
        data = {
            "text_prompt": text_prompt,
            "image": self._encode_image(image_path),
            "box_threshold": box_threshold,
            "text_threshold": text_threshold
        }

        # Send request
        response = requests.post(self.grounding_dino_url, json=data)

        # Check response
        if response.status_code != 200:
            error_message = response.json().get('error', 'Unknown error')
            raise Exception(f"API request failed: {error_message}")

        return response.json()

    def segment_objects(self, image_path, boxes):
        """
        Segment objects using SAM2

        Args:
            image_path: Path to the image file
            boxes: List of bounding boxes in xyxy format

        Returns:
            Segmentation results from SAM2
        """
        # Prepare request data
        data = {
            "boxes": boxes,
            "image": self._encode_image(image_path)
        }

        # Send request
        response = requests.post(self.sam2_url, json=data)

        # Check response
        if response.status_code != 200:
            error_message = response.json().get('error', 'Unknown error')
            raise Exception(f"API request failed: {error_message}")

        return response.json()

    def detect_and_segment(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Combined detection and segmentation using both models

        Args:
            image_path: Path to the image file
            text_prompt: Text prompt for object detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text

        Returns:
            Combined results from both models
        """
        # Ensure text prompt ends with periods
        if not all(term.strip().endswith('.') for term in text_prompt.split()):
            text_prompt = ' '.join([term.strip() + '.' if not term.strip().endswith('.') else term.strip()
                                    for term in text_prompt.split()])

        # Prepare request data
        data = {
            "text_prompt": text_prompt,
            "image": self._encode_image(image_path),
            "box_threshold": box_threshold,
            "text_threshold": text_threshold
        }

        # Send request
        response = requests.post(self.combined_url, json=data)

        # Check response
        if response.status_code != 200:
            error_message = response.json().get('error', 'Unknown error')
            raise Exception(f"API request failed: {error_message}")

        return response.json()

    def visualize_results(self, image_path, results, output_path=None):
        """
        Visualize detection and segmentation results

        Args:
            image_path: Path to the original image
            results: Results from detect_and_segment method
            output_path: Path to save the visualized image (optional)

        Returns:
            Annotated image
        """
        # Load image
        img = cv2.imread(image_path)

        # Extract results
        boxes = np.array(results["boxes"])
        labels = results["labels"]
        confidences = results["confidences"]
        masks_rle = results["masks_rle"]

        # Convert RLE masks to binary masks
        masks = []
        for rle in masks_rle:
            mask = mask_util.decode(rle)
            masks.append(mask)

        masks = np.array(masks).astype(bool)

        # Create detections object
        class_ids = np.array(list(range(len(labels))))
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks,
            class_id=class_ids
        )

        # Create formatted labels
        formatted_labels = [
            f"{label} {confidence:.2f}"
            for label, confidence in zip(labels, confidences)
        ]

        # Annotate image
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=formatted_labels)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # Save if output path is provided
        if output_path:
            cv2.imwrite(output_path, annotated_frame)

        return annotated_frame

    def decode_rle_masks(self, masks_rle, height, width):
        """
        Decode RLE masks to binary masks

        Args:
            masks_rle: List of RLE encoded masks
            height: Image height
            width: Image width

        Returns:
            List of binary masks
        """
        masks = []
        for rle in masks_rle:
            mask = mask_util.decode(rle)
            masks.append(mask)

        return np.array(masks).astype(bool)

# Usage example
def main():
    # Initialize client
    client = GroundedSAM2Client()

    # Path to test image
    image_path = "/media/yons/WIN10/prog/seamless_communication/Qwen25_VL/cookbooks/assets/spatial_understanding/r3.jpg"

    # Text prompt for object detection
    text_prompt = "person. car. dog."

    # Step 1: Detect objects
    print("Detecting objects...")
    detection_results = client.detect_objects(image_path, text_prompt)
    print(f"Found {len(detection_results['boxes'])} objects")

    # Step 2: Segment objects
    print(f"boxes:{detection_results['boxes']}")

    # Step 2: Segment objects
    print("Segmenting objects...")
    # detection_results_ = detection_results['boxes']
    detection_results_ = [[918.216552734375, 509.0528259277344, 1115.3170166015625, 645.4674072265625]]
    segmentation_results = client.segment_objects(image_path, detection_results_)
    print(f"Generated {len(segmentation_results['masks_rle'])} segmentation masks")
    # print("Segmenting objects...")
    # segmentation_results = client.segment_objects(image_path, detection_results['boxes'])
    # print(f"Generated {len(segmentation_results['masks_rle'])} segmentation masks")

    # Alternative: Combined detection and segmentation
    print("Performing combined detection and segmentation...")
    combined_results = client.detect_and_segment(image_path, text_prompt)
    # print(f"combined_results:{combined_results}")
    # Visualize results
    print("Visualizing results...")
    output_path = "output_visualization.jpg"
    annotated_image = client.visualize_results(image_path, combined_results, output_path)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()