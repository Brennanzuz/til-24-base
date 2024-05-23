from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    CLIPProcessor,
    CLIPModel,
)


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

# Assume the rest of your model training setup is here
# ....

# After training:
detr_model.save_pretrained(detr_model_path)
detr_processor.save_pretrained(detr_model_path)

clip_model.save_pretrained(clip_model_path)
clip_processor.save_pretrained(clip_model_path)