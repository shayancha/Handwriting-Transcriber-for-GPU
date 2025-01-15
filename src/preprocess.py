import cv2
import numpy as np


class Preprocessor:
    def __init__(self,
                 target_img_size: (int, int),
                 shear_factors: np.ndarray,
                 padding: int = 0,
                 automatic_binarization: bool = False,
                 data_augmentation: bool = False
                 ):
        self.target_img_size = target_img_size
        self.shear_factors = shear_factors
        self.padding = padding
        self.automatic_binarization = automatic_binarization
        # add data augmentation features later
        self.data_augmentation = data_augmentation

    def reduce_noise(self, img, ksize, sigma_x):
        blurred_img = cv2.GaussianBlur(img, ksize, sigma_x)
        return blurred_img

    def bilateral_filter(self, img, d, sigma_color, sigma_space):
        filtered_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        return filtered_img

    def binarize(self, img, gray_level=0):
        if self.automatic_binarization:
            _, new_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            if gray_level is None:
                raise ValueError("gray_level must be provided when automatic_binarization is false")
            _, new_img = cv2.threshold(img, gray_level, 255, cv2.THRESH_BINARY)

        return new_img

    def pad_horizontally(self, img):
        old_h, old_w = img.shape[:2]
        new_h = old_h
        new_w = 2 * self.padding + old_w

        padded_img = np.ones((new_h, new_w), dtype=np.uint8) * 255;

        padded_img[:, self.padding:self.padding + old_w] = img

        return padded_img

    def resize_with_padding(self, img):
        old_h, old_w = img.shape[:2]
        target_h, target_w = self.target_img_size

        scale = min(target_w / old_w, target_h / old_h)
        new_w = int(old_w * scale)
        new_h = int(old_h * scale)

        resized_img = cv2.resize(img, (new_w, new_h))

        padded_img = np.ones((target_h, target_w), dtype=np.uint8)*255

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        return padded_img

    def deslant(self, img):
        max_score = -1
        best_shear_factor = 0

        rows, cols = img.shape
        for alpha in self.shear_factors:
            shear_matrix = np.float32([[1, alpha, 0], [0, 1, 0]])
            sheared_img = cv2.warpAffine(img, shear_matrix, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=255)

            score = 0
            for col in range(cols):
                column_data = sheared_img[:, col]
                foreground_pixels = np.where(column_data == 0)[0]
                if len(foreground_pixels) > 0:
                    h_alpha = len(foreground_pixels)
                    delta_y_alpha = foreground_pixels[-1] - foreground_pixels[0]
                    if h_alpha == delta_y_alpha:
                        score += h_alpha ** 2

            if score > max_score:
                max_score = score
                best_shear_factor = alpha

        shear_matrix = np.float32([[1, best_shear_factor, 0], [0, 1, 0]])
        desheared_img = cv2.warpAffine(img, shear_matrix, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=255)

        return desheared_img

    def process_img(self, img, gray_level=None):
        new_img = self.binarize(img, gray_level)
        new_img = self.pad_horizontally(new_img)
        new_img = self.deslant(new_img)
        new_img = self.resize_with_padding(new_img)
        return new_img


def main():
    image_path_folder = "../large_iam_lines/dataset/"
    preprocessed_image_folder = "large_iam_lines/preprocessed_dataset/"
    metadata_file = "../large_iam_lines/metadata/sentences.txt"
    padding = 50
    max_img_height = 342
    # should the width depend on padding??
    max_img_width = 2270 + padding
    shear_factors = np.linspace(-0.5, 0.5, 50)

    # preprocessor = Preprocessor((max_img_height, max_img_width), shear_factors, padding)
    # with open(metadata_file, "r") as file:
    #     for line_num, line in enumerate(file, start=1):
    #         if not line.startswith("#"):
    #             image_id = line.split()[0]
    #             gray_level = int(line.split()[3])
    #
    #             original_img = cv2.imread(image_path_folder + image_id + ".png", cv2.IMREAD_GRAYSCALE)
    #             preprocessed_img = preprocessor.process_img(original_img, gray_level)
    #             cv2.imwrite(preprocessed_image_folder + image_id + ".png", preprocessed_img)
    #
    #             print(f"img number: {line_num }, image_id: {image_id}")

    preprocessor = Preprocessor((max_img_height, max_img_width), shear_factors, padding, automatic_binarization=True)
    original_img = cv2.imread("../large_iam_lines/dataset/a01-000u-s00-00.png", cv2.IMREAD_GRAYSCALE)
    preprocessed_img = preprocessor.process_img(original_img)
    cv2.imwrite("../selfmade_inference_dataset/preprocessed/selfMade_0.png", preprocessed_img)


if __name__ == '__main__':
    main()

