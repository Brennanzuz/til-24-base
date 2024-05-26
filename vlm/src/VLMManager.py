# from typing import List
# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import (
#     AutoProcessor,
#     AutoModelForZeroShotObjectDetection
# )
# import numpy as np
# import io
# from PIL import Image
# import torch
# import os
# from pathlib import Path
import os
import logging
import json

import pandas as pd
import numpy as np

from PIL import Image, ImageDraw
import albumentations

from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, AutoProcessor, AutoModelForZeroShotObjectDetection,  pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from torchmetrics.detection import MeanAveragePrecision


class VLMManager:
    def __init__(self):
        pass 

    def identify(self, image: bytes, caption: str) -> List[int]:
        model_path = "local/custom_owl_vit_1"
        detector = pipeline(model=model_path, task="zero-shot-object-detection")

        prediction = detector(
            image,
            candidate_labels=[caption],
            threshold=0.10,
            top_k=8
        )

        draw = ImageDraw.Draw(image)

        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        
        return [xmin, ymin, xmax, ymax]