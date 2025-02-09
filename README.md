# Handwriting Transcriber for GPU
Repo for running trainings from TranscribeThis! on a GPU instance

## Setup on GPU Instance
1. Clone this repo
2. pip install torch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. pip install other libraries: `pip install transformers opencv-python numpy`
4. transfer files large_iam_lines.zip from local computer over to GPU instance

## Run Training
`python src/main.py --mode train --img_dir "large_iam_lines/dataset/" --metadata "large_iam_lines/metadata/sentences.txt"`

