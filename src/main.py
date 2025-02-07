import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from dataloader import DataLoader,HTRDataset, collate
from model import HTRModel, ModelTrainer, DecoderType
from preprocess import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run the HTR model.")
    parser.add_argument("--mode", choices=["train", "validate", "infer"], default="infer", help="Mode to runt the script")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--decoder", choices=["bestpath", "beamsearch", "wordbeamsearch"], default="bestpath", help="Decoding method")
    parser.add_argument("--img_file", type=str, default=None, help="Path to an image for inference")
    parser.add_argument("--line_mode", action="store_true", help="Use line mode (for text lines)")
    parser.add_argument("--early-stopping", type=int, default=10, help="Early stopping epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    return parser.parse_args()


def train(args, char_list, decoder):
    preprocessor = Preprocessor(target_img_size=(342, 2320), shear_factors=np.linspace(-0.5, 0.5, 50), padding=50)

    dataloader = DataLoader(data_dir=Path(args.data_dir), preprocessor=preprocessor, batch_size=args.batch_size, augment=True)

    dataloader.train_set()

    with open("../wordCharList.txt", "w") as f:
        word_chars = "".join([char for char in char_list if char.strip()])
        f.write(word_chars)

    with open("../corpus.txt", "w") as f:
        words = " ".join(dataloader.train_words + dataloader.validation_words)
        f.write(words)

    train_dataset = HTRDataset(dataloader, augment=True)
    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)

    dataloader.validation_set()
    val_dataset = HTRDataset(dataloader, augment=False)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = (args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = HTRModel(char_list=char_list, decoder_type=decoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = ModelTrainer(model=model, char_list=char_list, device=device)

    trainer.train(train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, num_epochs=args.epochs)


def validate(args, char_list, decoder, model_dir="../models/best_model.pth"):
    preprocessor = Preprocessor(target_img_size=(342, 2320), shear_factors=np.linspace(-0.5, 0.5, 50), padding=50)
    dataloader = DataLoader(data_dir=Path(args.data_dir), preprocessor=preprocessor, batch_size=args.batch_size)

    dataloader.validation_set()
    val_dataset = HTRDataset(dataloader)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = (args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = HTRModel(char_list=char_list, decoder_type=decoder)
    model.load_state_dict(torch.load(model_dir))
    model.to(device)
    model.eval()

    total_loss = 0
    trainer = ModelTrainer(model=model, char_list=char_list, device=device)
    for batch_num, batch in enumerate(val_loader):
        print(f"validating batch {batch_num}")
        with torch.no_grad():
            images = batch["images"].to(device)
            ground_truths = batch["ground_truths"]
            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            targets, target_lengths = trainer.encode_targets(ground_truths)
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(device)
            loss = trainer.ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(val_loader)}")


def decode_text(indices, char_list):
    return "".join([char_list[idx] if idx < len(char_list) else "" for idx in indices])


def infer(args, char_list, decoder):
    device = (args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    # model = HTRModel(char_list=char_list, decoder_type=DecoderType.WordBeamSearch)
    model = HTRModel(char_list=char_list, decoder_type=decoder)

    model_path = "../models/best_model.pth"
    # if decoder == DecoderType.WordBeamSearch:
    #     model_path = "../models_wordbeamsearch/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    model.to(device)
    model.eval()

    preprocessor = Preprocessor(target_img_size=(342, 2320), shear_factors=np.linspace(-0.5, 0.5, 50), padding=50, automatic_binarization=True)
    img_path = Path(args.img_file)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = preprocessor.process_img(img)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        logits = model(img_tensor)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        trainer = ModelTrainer(model=model, char_list=char_list, device=device)
        prediction = trainer.decode_predictions(log_probs)
        # if decoder == DecoderType.WordBeamSearch:
        #     decoded_text = decode_text(prediction[0], char_list)
        #     print(f"Predicted Text: {decoded_text}")
        # else:
        print(f"Predicted Text: {prediction[0]}")


def load_char_list():
    with open("../wordCharList.txt", "r") as f:
        return list(f.read().strip())


def main():
    args = parse_args()

    char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-'\":{}[]/\_+=`~;#@$%^&*() ")

    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder = decoder_mapping[args.decoder]

    if args.mode == "train":
        train(args, char_list, decoder)
    elif args.mode == "validate":
        validate(args, char_list, decoder)
    elif args.mode == "infer":
        if args.img_file is None:
            print("Please specify an image  file for inference using --img-file")
        else:
            infer(args, char_list, decoder)


if __name__ == "__main__":
    main()
