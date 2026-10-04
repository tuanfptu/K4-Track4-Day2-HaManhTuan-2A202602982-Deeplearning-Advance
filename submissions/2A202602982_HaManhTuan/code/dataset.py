"""DeepWeeds fold-0 data loading and transforms."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_split(labels_dir, fold=0):
    if fold != 0:
        raise ValueError("Lab requires author's fold 0")
    return tuple(pd.read_csv(Path(labels_dir) / f"{part}_subset{fold}.csv") for part in ("train", "val", "test"))

def check_split(train_df, val_df, test_df, images_dir):
    groups = {"train": train_df, "val": val_df, "test": test_df}
    names = {}
    for part, frame in groups.items():
        if not {"Filename", "Label"}.issubset(frame.columns):
            raise ValueError(f"{part}: missing columns")
        if frame.Filename.isna().any() or frame.Filename.duplicated().any():
            raise ValueError(f"{part}: duplicate/null filename")
        if not np.isin(frame.Label.to_numpy(), np.arange(NUM_CLASSES)).all():
            raise ValueError(f"{part}: invalid labels")
        names[part] = set(frame.Filename)
    overlap = {f"{a}_{b}": len(names[a] & names[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    union = set.union(*names.values())
    if any(overlap.values()) or len(union) != 17509:
        raise ValueError("Fold 0 overlapping or incomplete")
    missing = [n for n in union if not (Path(images_dir) / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing; first={missing[0]}")
    return {"n": {k: len(v) for k, v in groups.items()}, "overlap": overlap,
            "per_class": {k: {str(c): int((v.Label == c).sum()) for c in range(NUM_CLASSES)} for k, v in groups.items()}}

def build_transforms(train, img_size=224, aug="basic"):
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ops = [T.RandomResizedCrop(img_size, scale=(0.7, 1.0)), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(0.2, 0.2, 0.2, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(2, 7))
    elif aug != "basic":
        raise ValueError(f"Unknown aug: {aug}")
    return T.Compose([*ops, T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.root / row.Filename) as im:
            image = im.convert("RGB")
        return (self.transform(image) if self.transform else image, int(row.Label), str(row.Filename))

def _seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2):
    if sampler not in (None, "balanced") or (sampler and not train):
        raise ValueError(f"Invalid sampler: {sampler}")
    weighted = None
    if sampler == "balanced":
        counts = df.Label.value_counts()
        weighted = WeightedRandomSampler([1 / counts[int(y)] for y in df.Label], len(df), replacement=True)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=bool(train and weighted is None), sampler=weighted,
                      drop_last=bool(train and len(df) > batch_size), num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
                      persistent_workers=num_workers > 0)
