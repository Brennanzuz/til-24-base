from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import Owlv2Processor, Owlv2ForObjectDetection
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
        self.model_directory = os.getenv("MODEL_PATH", Path("vlm/src/models").absolute()) # For local tests
        # self.model_directory = os.getenv("MODEL_PATH", Path("models").absolute())
        self.model_filename = "owlv2.pth"  # Specify your model filename here

        # Full path to your model files
        self.model_path = os.path.join(self.model_directory, self.model_filename)

        # Load the models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_path, device_map=self.device
        )
        self.processor = Owlv2Processor.from_pretrained(self.model_path, device_map=self.device)

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
            boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
            highest_score = 0
            best_box = [0.0, 0.0, 0.0, 0.0]
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
                if score > highest_score:
                    highest_score = score
                    best_box = box
                    
        print(f"Best box: {best_box} with score {highest_score}")
        return best_box