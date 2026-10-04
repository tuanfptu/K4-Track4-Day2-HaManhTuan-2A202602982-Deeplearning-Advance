"""Download and verify the original DeepWeeds fold-0 dataset."""
import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

IMAGE_URL = "https://zenodo.org/records/7939060/files/images.zip?download=1"
IMAGE_MD5 = "b7b30f96d466fba86016aa5a26606e0f"
LABEL_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/"
LABELS = ("labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv")


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target):
    temporary = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    temporary.replace(target)


def prepare(data):
    data = Path(data)
    labels = data / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for name in LABELS:
        if not (labels / name).is_file():
            download(LABEL_URL + name, labels / name)
    images = data / "images"
    if len(list(images.glob("*.jpg"))) == 17509:
        print("Images already extracted: 17,509")
        return
    archive = data / "images.zip"
    if not archive.is_file() or md5_file(archive) != IMAGE_MD5:
        download(IMAGE_URL, archive)
    digest = md5_file(archive)
    if digest != IMAGE_MD5:
        raise ValueError(f"Image archive MD5 mismatch: {digest}")
    images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            destination = (images / item.filename).resolve()
            if not destination.is_relative_to(images.resolve()):
                raise ValueError(f"Unsafe zip path: {item.filename}")
        source.extractall(images)
    count = len(list(images.glob("*.jpg")))
    if count != 17509:
        raise ValueError(f"Expected 17,509 images, found {count}")
    print(f"Verified {count} images and original labels at {data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    prepare(parser.parse_args().data)
