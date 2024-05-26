import torch
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

checkpoint = "google/owlvit-base-patch32"
model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

model_path = "vlm_model.pth"

# # Data augmentation and normalization for training
# # Just normalization for validation
# dataset_transforms = {
#     "train": transforms.Compose([
#         Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
#         CenterCrop(224),
#         ConvertImageDtype(torch.float),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     "val": transforms.Compose([
#         Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
#         CenterCrop(224),
#         ConvertImageDtype(torch.float),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
# }

# # Training
# # load datasets, a folder of images, and a jsonl file with captions and bounding boxes
# dataset = datasets.ImageFolder("../../../advanced/images")

# num_train = len(dataset)
# num_val = int(0.2 * num_train)
# num_train -= num_val

# train_subset, val_subset = random_split(dataset, [num_train, num_val])

# train_subset.dataset.transform = dataset_transforms['train']
# val_subset.dataset.transform = dataset_transforms['val']

# dataloaders = {
#     'train': torch.utils.data.DataLoader(train_subset.dataset, batch_size=4, shuffle=True, num_workers=4),
#     'val': torch.utils.data.DataLoader(val_subset.dataset, batch_size=4, shuffle=False, num_workers=4)
# }

# dataset_sizes = {
#     'train': num_train,
#     'val': num_val
# }

# After training:
model.save_pretrained(model_path)
processor.save_pretrained(model_path)