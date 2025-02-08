from pathlib import Path
import numpy as np
from preprocess import Preprocessor
from dataset import DataLoader
import word_beam_search

# preprocessor = Preprocessor(
#     target_img_size=(342, 2270+50),
#     shear_factors=np.linspace(-0.5, 0.5, 50),
#     padding=50
# )
#
# data_dir = Path("large_iam_lines")
# dataloader = DataLoader(
#     data_dir=data_dir,
#     preprocessor=preprocessor,
#     batch_size=4
# )
#
# dataloader.train_set()
# while dataloader.has_next():
#     batch = dataloader.get_next_batch()
#     print("Batch size:", batch["batch_size"])
#     print("First ground truth:", batch["ground_truths"][0])

print(word_beam_search)
