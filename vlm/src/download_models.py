import requests
import torch
from ultralytics import YOLO
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import Owlv2Processor, Owlv2ForObjectDetection

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", force_download=True)
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", force_download=True)
model_path = "owlv2.pth"

processor.save_pretrained(model_path)
model.save_pretrained(model_path)