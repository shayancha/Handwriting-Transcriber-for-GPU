# Handwriting Transcriber for GPU
Repo for running trainings from TranscribeThis! on a GPU instance

## Setup on GPU Instance
1. Clone this repo with `git clone https://github.com/shayancha/Handwriting-Transcriber-for-GPU.git`
2. Switch to this branch with `git checkout trocr`
4. Install torch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
5. Install other libraries: `pip install transformers opencv-python numpy tf-keras transformers[torch]`
6. Transfer files large_iam_lines.zip from local computer over to GPU instance using `scp -i ~/.ssh/my-key.pem ~/path/to/local_file ubuntu@instance-ip-address:~/path/to/destination` (note: it must be placed at the root of the Handwriting-Transcriber-for-GPU directory)
7. Extract word image data: `unzip large_iam_lines.zip`

## Run Training
`python src/main.py --mode train --img_dir "large_iam_lines/dataset/" --metadata "large_iam_lines/metadata/sentences.txt"`

