"""Shared preprocessing module for the ENGE707 cat/dog facial feature classifier.

Every training, evaluation and prediction script imports from this file, so that images are
always processed in exactly the same way. It defines:
  - project paths and fixed settings
  - the class-to-integer label mapping
  - the image transforms (one for training, one for validation and test)
  - the Dataset class and DataLoader helpers

Design decisions (justified in report Task 2):
  - Centre-crop to a square on the shortest side, then resize to 128 x 128.
    This removes the class-specific crop shape (e.g. wide mouth crops) as a shortcut.
  - Normalise with the ImageNet mean and standard deviation, as expected by pretrained models.
    These are fixed published values, so nothing is computed from validation or test data.
  - Augmentation is applied to training images only, and avoids strong zoom because
    framing is part of what separates Mouth crops from Nose crops.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

# ---------------------------------------------------------------------------
# Paths and fixed settings
# ---------------------------------------------------------------------------
def _find_project_root():
    """Walks up from this file's folder until it finds the folder containing data/.
    This lets scripts and notebooks run from the project folder or any subfolder (e.g. src/)."""
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        if (folder / "data").is_dir():
            return folder
    raise FileNotFoundError(f"Could not find a 'data' folder in {here} or any folder above it.")


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = DATA_DIR / "train_labels.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPLIT_DIR = OUTPUT_DIR / "splits"
FIG_DIR = OUTPUT_DIR / "figures"
QUALITY_CSV = OUTPUT_DIR / "quality_metrics.csv"       # produced by 03_data_quality.py
CLASS_MAP_JSON = SPLIT_DIR / "class_mapping.json"

SEED = 42
IMAGE_SIZE = 128
VAL_FRACTION = 0.20
MIN_SIDE = 16             # training images with a shorter side than this are excluded (Task 1.6)

# ImageNet statistics, used because pretrained backbones were trained with them
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fixed class order. The index of each class is its integer label.
CLASS_NAMES = ["CatEye", "DogEye", "CatFace", "DogFace", "CatMouth", "DogMouth", "CatNose", "DogNose"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def relative_path(path):
    """Path relative to the project folder, as stored in the split CSVs (keeps the repo portable)."""
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def save_class_mapping(path=CLASS_MAP_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"class_to_idx": CLASS_TO_IDX, "idx_to_class": IDX_TO_CLASS}, indent=2))


def get_device():
    """Apple silicon GPU (mps) if available, then CUDA, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class SquareCenterCrop(torch.nn.Module):
    """Crops the largest centred square from an image (side = shortest side).
    Unlike torchvision's CenterCrop, the crop size adapts to each image."""

    def forward(self, img):
        w, h = F.get_size(img)[::-1]          # get_size returns [height, width]
        side = min(w, h)
        return F.center_crop(img, [side, side])


def _to_normalised_tensor():
    return [
        T.ToImage(),                                   # PIL image -> uint8 tensor (C, H, W)
        T.ToDtype(torch.float32, scale=True),          # rescale pixel values from 0-255 to 0-1
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def get_eval_transform(size=IMAGE_SIZE):
    """Deterministic preprocessing for validation and test images (and the core of training)."""
    return T.Compose([
        SquareCenterCrop(),
        T.Resize((size, size), antialias=True),
        *_to_normalised_tensor(),
    ])


def get_train_transform(size=IMAGE_SIZE):
    """Evaluation preprocessing plus random augmentation. Used for training images only."""
    return T.Compose([
        SquareCenterCrop(),
        # Mild zoom only: keeps 85-100% of the square and never changes its shape (ratio fixed at 1)
        T.RandomResizedCrop((size, size), scale=(0.85, 1.0), ratio=(1.0, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        *_to_normalised_tensor(),
    ])


def denormalise(tensor):
    """Reverses the normalisation so a tensor can be displayed as an image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ---------------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------------
class FacialFeatureDataset(Dataset):
    """Reads images listed in a split CSV (columns: image_id, path, and label for labelled splits).

    Returns (image_tensor, label) for labelled data, or (image_tensor, image_id) for test data.
    Images are loaded from disk one at a time, so the full dataset never sits in memory.
    """

    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path, dtype={"image_id": str})
        self.transform = transform
        self.has_labels = "label" in self.df.columns
        self.paths = [PROJECT_ROOT / p for p in self.df["path"]]   # CSV paths are relative to the project
        self.targets = self.df["label"].map(CLASS_TO_IDX).tolist() if self.has_labels else None
        self.ids = self.df["image_id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        x = self.transform(img)
        return (x, self.targets[i]) if self.has_labels else (x, self.ids[i])


def get_loaders(batch_size=128, num_workers=4, size=IMAGE_SIZE):
    """Builds the training, validation and test DataLoaders from the saved split files.

    Training data is shuffled and augmented; validation and test use the same deterministic
    transform and keep a fixed order. The test order follows sample_submission.csv.
    """
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  persistent_workers=num_workers > 0)
    train_ds = FacialFeatureDataset(SPLIT_DIR / "train.csv", get_train_transform(size))
    val_ds = FacialFeatureDataset(SPLIT_DIR / "val.csv", get_eval_transform(size))
    test_ds = FacialFeatureDataset(SPLIT_DIR / "test.csv", get_eval_transform(size))
    generator = torch.Generator().manual_seed(SEED)       # reproducible shuffling order
    return (DataLoader(train_ds, shuffle=True, generator=generator, **common),
            DataLoader(val_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common))