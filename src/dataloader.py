import random
from typing import List, Dict, Union
import cv2
import numpy as np
from pathlib import Path
from preprocess import Preprocessor
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# maybe add custom Sample and Batch types
class DataLoader:
    def __init__(self, data_dir: Path, preprocessor: Preprocessor, batch_size: int, data_split: float = 0.8, augment: bool = False):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.current_index = 0
        self.augment = augment

        self.augmentations = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-10, 10), shear=(-10, 10), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.Perspective(scale=(0.05, 0.15), p=0.2),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])

        self.samples = self._load_samples()

        split_index = int(len(self.samples) * data_split)
        self.train_samples = self.samples[:split_index]
        self.validation_samples = self.samples[split_index:]

        self.train_words = self._extract_words(self.train_samples)
        self.validation_words = self._extract_words(self.validation_samples)

        # might not need mode to check if mode is correct for augmentation
        self.mode = "train"
        self.train_set()

    def _load_samples(self) -> List[Dict[str, Union[Path, str]]]:
        samples = []
        gt_file = self.data_dir / "metadata/sentences.txt"

        with open(gt_file, "r") as file:
            for line in file:
                if not line.startswith("#"):
                    img_file = line.split()[0]
                    img_file_path = (self.data_dir / "dataset" / img_file).with_suffix(".png")
                    gray_level = int(line.split()[3])
                    ground_truth = line.split()[-1].replace("|", " ")
                    samples.append({"file_path": img_file_path,
                                    "gray_level": gray_level,
                                    "ground_truth": ground_truth})

        return samples

    def _extract_words(self, samples: List[Dict[str, Union[Path, str]]]) -> List[str]:
        words = []
        for sample in samples:
            ground_truth = sample["ground_truth"]
            words.extend(ground_truth.split())
        return words

    def train_set(self) -> None:
        self.mode = "train"
        self.samples = self.train_samples
        self.current_index = 0
        random.shuffle(self.samples)

    def validation_set(self) -> None:
        self.mode = "validate"
        self.samples = self.validation_samples
        self.current_index = 0
        # probably do not need to shuffle validation set
        # random.shuffle(self.samples)

    def has_next(self) -> bool:
        return self.current_index < len(self.samples)

    def get_next_batch(self) -> Dict[str, Union[List[np.ndarray], List[str], int]]:
        batch_range = range(self.current_index, min(self.current_index + self.batch_size, len(self.samples)))

        processed_imgs = []
        ground_truths = []

        for index in batch_range:
            sample = self.samples[index]
            file_path: Path = sample["file_path"]
            gray_level: str = sample["gray_level"]
            ground_truth: str = sample["ground_truth"]

            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

            if img is not None:
                processed_img = self.preprocessor.process_img(img, gray_level)

                print(f"Augmenting batch: {self.augment}")
                if self.augment and self.mode == "train":
                    processed_img = self.apply_augmentation(processed_img)

                processed_imgs.append(processed_img)
                ground_truths.append(ground_truth)
            else:
                print(f"Warning: Could not load image at {file_path}")

        self.current_index += self.batch_size

        return {
            "images": processed_imgs,
            "ground_truths": ground_truths,
            "batch_size": len(processed_imgs)
        }

    def apply_augmentation(self, img):
        img = np.expand_dims(img, axis=-1)
        img = self.augmentations(image=img)["image"]

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        return img.squeeze(0)


class HTRDataset(Dataset):
    def __init__(self, dataloader: DataLoader, augment: bool = False):
        self.dataloader = dataloader
        self.samples = self.dataloader.samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        file_path = sample["file_path"]
        gray_level = sample["gray_level"]
        ground_truth = sample["ground_truth"]

        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        img = self.dataloader.preprocessor.process_img(img, gray_level)

        if self.dataloader.augment:
            img = self.dataloader.apply_augmentation(img)

        img = torch.from_numpy(img).clone().detach().float().unsqueeze(0) / 255.0

        return {"image": img, "ground_truth": ground_truth}


def collate(batch):
    images = torch.stack([item["image"] for item in batch])
    ground_truths = [item["ground_truth"] for item in batch]
    return {"images": images, "ground_truths":  ground_truths}
