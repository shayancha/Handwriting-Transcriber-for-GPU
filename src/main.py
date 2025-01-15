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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--decoder", choices=["bestpath, beamsearch, wordbeamsearch"], default="bestpath", help="Decoding method")
    parser.add_argument("--img_file", type=str, default=None, help="Path to an image for inference")
    parser.add_argument("--line_mode", action="store_true", help="Use line mode (for text lines)")
    parser.add_argument("--early-stopping", type=int, default=10, help="Early stopping epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    return parser.parse_args()


def train(args, char_list):
    preprocessor = Preprocessor(target_img_size=(342, 2320), shear_factors=np.linspace(-0.5, 0.5, 50), padding=50)

    dataloader = DataLoader(data_dir=Path(args.data_dir), preprocessor=preprocessor, batch_size=args.batch_size)

    dataloader.train_set()
    train_dataset = HTRDataset(dataloader)
    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)

    dataloader.validation_set()
    val_dataset = HTRDataset(dataloader)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = (args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = HTRModel(char_list=char_list, decoder_type=DecoderType.BestPath).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = ModelTrainer(model=model, char_list=char_list, device=device)

    trainer.train(train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, num_epochs=args.epochs)

    # add validation to each training epoch - should switch to validation dataset!
    # a val_loader should get passed to train()


def validate(args, char_list, model_dir="../models/best_model.pth"):
    preprocessor = Preprocessor(target_img_size=(342, 2320), shear_factors=np.linspace(-0.5, 0.5, 50), padding=50)
    dataloader = DataLoader(data_dir=Path(args.data_dir), preprocessor=preprocessor, batch_size=args.batch_size)

    dataloader.validation_set()
    val_dataset = HTRDataset(dataloader)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = (args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = HTRModel(char_list=char_list, decoder_type=DecoderType.BestPath)
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


def infer(args, char_list):
    device = (args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = HTRModel(char_list=char_list, decoder_type=DecoderType.BestPath)

    # use next line when using cpu
    model.load_state_dict(torch.load("../models/best_model.pth", map_location=torch.device(device)))
    # model.load_state_dict(torch.load("models/best_model.pth"))

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
        print(f"Predicted Text: {prediction[0]}")


def main():
    args = parse_args()
    char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-'\":{}[]/\_+=`~;#@$%^&*() ")

    if args.mode == "train":
        train(args, char_list)
    elif args.mode == "validate":
        validate(args, char_list)
    elif args.mode == "infer":
        if args.img_file is None:
            print("Please specify an image  file for inference using --img-file")
        else:
            infer(args, char_list)


if __name__ == "__main__":
    main()