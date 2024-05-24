from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    CLIPProcessor,
    CLIPModel,
)
import numpy as np
import io
from PIL import Image
import torch
import os
from pathlib import Path

class VLMManager:
    def __init__(self):
        # initialize the model here
        # Fetch the model directory from the environment variable
        self.model_directory = os.getenv("MODEL_PATH", Path("models").absolute())
        self.detr_model_filename = "detr_model.pth"  # Specify your model filename here
        self.clip_model_filename = "clip_model.pth"  # Specify your model filename here

        # Full path to your model files
        self.detr_model_path = os.path.join(self.model_directory, self.detr_model_filename)
        self.clip_model_path = os.path.join(self.model_directory, self.clip_model_filename)

        # Load the models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detr_model = AutoModelForObjectDetection.from_pretrained(
            self.detr_model_path, device_map=self.device
        )
        self.detr_processor = AutoImageProcessor.from_pretrained(self.detr_model_path, device_map=self.device)

        self.clip_model = CLIPModel.from_pretrained(self.clip_model_path, device_map=self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_path, device_map=self.device)

    def detect_objects(self, image):
        """Detects objects in an image using DETR.

        Args:
            image (Image): The image to detect objects in.

        Returns:
            list: A list of bounding boxes for the detected objects.
        """
        with torch.no_grad():
            inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.detr_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.detr_processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]
        return results["boxes"]


    def object_images(self, image, boxes):
        """Extracts images of objects from the original image.
        
        Args:
            image (Image): The original image.
            boxes (list): A list of bounding boxes for the objects.
            
        Returns:
            list: A list of images of the objects.
        """
        image_arr = np.array(image)
        all_images = []
        for box in boxes:
            # DETR returns top, left, bottom, right format
            x1, y1, x2, y2 = [int(val) for val in box]
            _image = image_arr[y1:y2, x1:x2]
            all_images.append(_image)
        return all_images


    def identify_target(self, query, images):
        """Identifies the most similar object to the query.
        
        Args:
            query (str): The text query.
            images (list): A list of images of objects.
            
        Returns:
            int: The index of the most similar object.
        """
        inputs = self.clip_processor(
            text=[query], images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        most_similar_idx = torch.argmax(logits_per_image, dim=0).item()
        return most_similar_idx

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        # image_bytes = base64.b64decode(image)
        im = Image.open(io.BytesIO(image))

        # detect object bounding boxes
        detected_objects = self.detect_objects(im)

        # get images of objects
        images = self.object_images(im, detected_objects)

        # identify target
        idx = self.identify_target(caption, images)

        # return bounding box of best match
        return [int(val) for val in detected_objects[idx].tolist()]
