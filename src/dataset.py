import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocess import TrOCRImageProcessor


class HandwritingDataset(Dataset):
    def __init__(self, img_dir, metadata_path, tokenizer, img_processor, max_target_length=128, augment=False):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.max_target_length = max_target_length
        self.augment = augment

        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-10, 10), shear=(-10, 10), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.Perspective(scale=(0.05, 0.15), p=0.2),
            ToTensorV2()
        ]) if augment else None

        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.samples = self._load_metadata(metadata_path)

    def _load_metadata(self, metadata_path):
        samples = []
        with open(metadata_path, "r") as file:
            for line in file:
                if not line.startswith("#"):
                    parts = line.split()
                    img_name = parts[0]
                    # might want to incorporate gray_level into samples later
                    gray_level = int(parts[3])
                    text_label = line.split()[-1].replace("|", " ")
                    samples.append((img_name, text_label))

        return samples

    def __len__(self):
        return len(self.samples)

    def adaptive_binarization(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def preprocess_image(self, img_path):
        img = self.img_processor.preprocess(img_path)

        img_np = img.permute(1, 2, 0).numpy()

        if self.augment:
            augmented = self.augmentation(image=img_np)
            img_np = augmented["image"]

        img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
        img = self.default_transform(img)

        return img

    def __getitem__(self, idx):
        img_name, text_label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name + ".png")

        img_tensor = self.preprocess_image(img_path)
        encoding = self.tokenizer(text_label, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_target_length)

        return {
            "pixel_values": img_tensor,
            "labels": encoding["input_ids"].squeeze()
        }

