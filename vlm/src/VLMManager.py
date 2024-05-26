from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
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
        # self.model_directory = os.getenv("MODEL_PATH", Path("vlm/src/models").absolute()) # For local tests
        self.model_directory = os.getenv("MODEL_PATH", Path("models").absolute())
        self.model_filename = "vlm_model.pth"  # Specify your model filename here

        # Full path to your model files
        self.model_path = os.path.join(self.model_directory, self.model_filename)

        # Load the models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path, device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, device_map=self.device)

    # def draw_bbox(self, image: bytes, bbox: List[int]) -> None:
    #     label = prediction["label"]
    #     score = prediction["score"]

    #     xmin, ymin, xmax, ymax = bbox.values()
    #     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    #     draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        # image_bytes = base64.b64decode(image)
        im = Image.open(io.BytesIO(image))

        # text prompts
        inputs = self.processor(text=[caption], images=im, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([im.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=0.1, target_sizes=target_sizes
            )[0]

        return results