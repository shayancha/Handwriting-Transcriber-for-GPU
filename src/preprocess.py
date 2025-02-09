from transformers import TrOCRProcessor
import cv2
from torchvision import transforms
import numpy as np


class TrOCRImageProcessor:
    def __init__(self, image_size=(384, 384), binarization=True, deslanting=True):
        self.image_size = image_size
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        # for faster inference and training use below
        # self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        self.binarization = binarization
        self.deslanting = deslanting

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.binarization:
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if self.deslanting:
            img = self.deslant(img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.transform(img)

        return img

    def deslant(self, img):
        max_score = -1
        best_shear_factor = 0
        rows, cols = img.shape
        shear_factors = np.linspace(-0.3, 0.3, 50)

        for alpha in shear_factors:
            shear_matrix = np.float32([[1, alpha, 0], [0, 1, 0]])
            sheared_img = cv2.warpAffine(img, shear_matrix, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=255)

            score = sum(len(np.where(sheared_img[:, col] == 0)[0]) ** 2 for col in range(cols))
            if score > max_score:
                max_score = score
                best_shear_factor = alpha

        shear_matrix = np.float32([[1, best_shear_factor, 0], [0, 1, 0]])
        return cv2.warpAffine(img, shear_matrix, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=255)
