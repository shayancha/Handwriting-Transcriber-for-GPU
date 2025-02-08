from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2

class HandwritingDataset(Dataset):
    def __init__(self, img_dir, metadata_path, tokenizer, img_processor, max_target_length=128):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.max_target_length = max_target_length

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

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform(img)

    def __getitem__(self, idx):
        img_name, text_label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name + ".png")

        img_tensor = self.preprocess_image(img_path)
        encoding = self.tokenizer(text_label, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_target_length)

        return {
            "pixel_values": img_tensor,
            "labels": encoding["input_ids"].squeeze()
        }

