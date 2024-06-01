import torch
from ultralytics import YOLO
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

# DETR
detr_checkpoint = "yolov8n.pt"
detr_model = YOLO(detr_checkpoint)

# CLIP
clip_checkpoint = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_checkpoint, force_download=True)
clip_processor = CLIPProcessor.from_pretrained(clip_checkpoint, force_download=True)

clip_model_path = "clip_model.pth"

# After training:
detr_model.save()

clip_model.save_pretrained(clip_model_path)
clip_processor.save_pretrained(clip_model_path)
