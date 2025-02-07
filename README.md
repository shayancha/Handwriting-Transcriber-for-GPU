# Handwriting Transcriber for GPU
Repo for running trainings from TranscribeThis! on a GPU instance

## Setup on GPU Instance
1. Clone CTCWordBeamSearch repo and follow installation steps: https://github.com/githubharald/CTCWordBeamSearch
2. pip install torch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
(replace cu118 with apropriate cuda version)
3. pip install other libraries: `pip install tqdm rich editdistance opencv-python numpy albumentations symspellpy`

## Run Training
`python main.py --mode train --data-dir ../large_iam_lines  --epochs 20 --decoder wordbeamsearch --device cuda`
