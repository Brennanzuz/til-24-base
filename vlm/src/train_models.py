import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    Trainer,
    TrainingArguments,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    CLIPProcessor,
    CLIPModel
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "FacebookAI/roberta-base"
)

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
config = model.config

# DETR
detr_checkpoint = "facebook/detr-resnet-50"
detr_model = AutoModelForObjectDetection.from_pretrained(detr_checkpoint, force_download=True)
detr_processor = AutoImageProcessor.from_pretrained(detr_checkpoint, force_download=True)

detr_model_path = "detr_model.pth"

# CLIP
clip_checkpoint = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_checkpoint, force_download=True)
clip_processor = CLIPProcessor.from_pretrained(clip_checkpoint, force_download=True)

clip_model_path = "clip_model.pth"

# After training:
detr_model.save_pretrained(detr_model_path)
detr_processor.save_pretrained(detr_model_path)

clip_model.save_pretrained(clip_model_path)
clip_processor.save_pretrained(clip_model_path)