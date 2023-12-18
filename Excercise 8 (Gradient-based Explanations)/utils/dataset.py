import json
import torch
import torchvision as tv
from pathlib import Path
from PIL import Image

# TEST_IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/f/ff/Junco_hyemalis_hyemalis-001.jpg'
TEST_IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/3/36/Dark-eyed_Junco%2C_Washington_State_02.jpg'
TEST_IMAGE_FILE = Path('test_image.jpg')

if not TEST_IMAGE_FILE.exists():
    torch.hub.download_url_to_file(TEST_IMAGE_URL, 'test_image.jpg')

IMAGENET_TRANSFORMS = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
])

IMAGENET_NORMALIZE = tv.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

with open('imagenet_idx_to_label.json', 'r') as f:
    IMAGENET_IDX_TO_LABEL = json.load(f)


def convert_idx_to_label(idx):
    return IMAGENET_IDX_TO_LABEL[str(idx)]


def load_test_image():
    image = Image.open(TEST_IMAGE_FILE)
    return image


def preprocess_image(image):
    image_preproc = IMAGENET_TRANSFORMS(image)
    return image_preproc


def normalize_image(image):
    image_norm= IMAGENET_NORMALIZE(image)
    return image_norm
