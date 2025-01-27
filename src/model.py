import math
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from dataloader import DataLoader
from torch.optim import Adam

import editdistance
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

import word_beam_search
from word_beam_search import WordBeamSearch


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class HTRModel(nn.Module):
    def __init__(self,
                 char_list: List[str],
                 rnn_hidden_size: int = 256,
                 num_rnn_layers: int = 2,
                 decoder_type: str = DecoderType.BestPath):
        super().__init__()
        self.char_list = char_list
        self.num_classes = len(char_list) + 1
        self.decoder_type = decoder_type

        if self.decoder_type == DecoderType.WordBeamSearch:
            with open("../wordCharList.txt", "r") as f:
                self.word_chars = f.read().strip()
            with open("../corpus.txt", "r") as f:
                self.corpus = f.read().strip()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, self.num_classes)

    def forward(self, x):
        x = self.cnn(x)

        batch_size, channels, height, width = x.size()
        assert height == 1, "height after CNN should be 1 --> adjust CNN layers"

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)

        x = self.fc(x)

        return x


class ModelTrainer:
    def __init__(self, model: HTRModel, char_list: List[str], device: str = "cuda"):
        self.model = model.to(device)
        self.char_list = char_list
        self.ctc_loss = nn.CTCLoss(blank=len(char_list), reduction="mean")
        self.device = device
        self.console = Console()

    def train(self, train_loader: TorchDataLoader, val_loader: TorchDataLoader, optimizer: Adam, num_epochs: int, save_dir="../models"):
        os.makedirs(save_dir, exist_ok=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        best_val_loss = float("inf")
        early_stopping_counter = 0

        for epoch_num, epoch in enumerate(range(num_epochs), start=1):
            self.console.print(f"\n[bold green]Epoch {epoch_num}/{num_epochs}[/bold green]")
            self.model.train()
            epoch_train_losses = []
            total_train_words = 0
            correct_train_words = 0

            with tqdm(train_loader, desc="Training", total=len(train_loader), leave=False) as tbar:
                for batch in train_loader:
                    start_time = time.time()

                    images = batch["images"].to(self.device)
                    ground_truths = batch["ground_truths"]

                    logits = self.model(images)
                    log_probs = F.log_softmax(logits, dim=2)

                    targets, target_lengths = self.encode_targets(ground_truths)
                    input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)
                    loss = self.ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_train_losses.append(loss.item())

                    predictions = self.decode_predictions(log_probs)
                    for gt, pred in zip(ground_truths, predictions):
                        total_train_words += 1
                        correct_train_words += 1 if gt == pred else 0

                    tbar.update(1)
                    tbar.set_postfix({"loss": loss.item(), "batch_time": f"{time.time()-start_time:.2f}s"})

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_accuracy = correct_train_words / total_train_words
            self.console.print(f"[bold blue]Training Loss:[/bold blue] {avg_train_loss:.4f}")
            self.console.print(f"[bold blue]Training Word Accuracy:[/bold blue] {train_accuracy:.4f}")

            val_loss = self.validate(val_loader)
            print(f"[bold yellow]Validation loss:[/bold yellow] {val_loss:.4f}")

            scheduler.step(val_loss)

            model_path = os.path.join(save_dir, f"model_epoch_{epoch_num}.pth")
            torch.save(self.model.state_dict(), model_path)
            self.console.print(f"[bold green]Model saved at {model_path}[/bold green]")

            if val_loss < best_val_loss:
                best_val_loss = avg_train_loss
                early_stopping_counter = 0
                best_model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                self.console.print(f"[bold green]Best model saved at {best_model_path} with loss: {best_val_loss:.4f}[/bold green]")
            else:
                early_stopping_counter+=1

            if early_stopping_counter >= 5:
                self.console.print("[bold red]Early stopping: No improvement in validation loss for 5 epochs[/bold red]")
                break

    def validate(self, val_loader: TorchDataLoader) -> float:
        self.model.eval()
        val_losses = []
        total_chars = 0
        total_char_errors = 0
        total_words = 0
        correct_words = 0

        with torch.no_grad():
            with tqdm(val_loader, desc="Validating", leave=False) as tbar:
                for batch in val_loader:
                    images = batch["images"].to(self.device)
                    ground_truths = batch["ground_truths"]

                    logits = self.model(images)
                    log_probs = F.log_softmax(logits, dim=2)

                    targets, target_lengths = self.encode_targets(ground_truths)
                    input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)

                    loss = self.ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
                    val_losses.append(loss.item())

                    predictions = self.decode_predictions(log_probs)

                    for gt, pred in zip(ground_truths, predictions):
                        total_chars+=len(gt)
                        total_char_errors += editdistance.eval(gt, pred)
                        total_words += 1
                        correct_words += 1 if gt == pred else 0

        char_error_rate = total_char_errors / total_chars
        word_accuracy = correct_words / total_words
        self.console.print(f"Validation - CER: {char_error_rate:.4f}, Word Accuracy: {word_accuracy:.4f}")
        return sum(val_losses) / len(val_losses)

    def infer(self, test_loader: DataLoader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["images"].to(self.device)

                logits = self.model(images)
                log_probs = F.log_softmax(logits, dim=2)

                pred = self.decode_predictions(log_probs)
                predictions.extend(pred)

        return predictions

    def encode_targets(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = []
        lengths = []
        for text in texts:
            encoded = [self.char_list.index(char) for char in text]
            targets.extend(encoded)
            lengths.append(len(encoded))

        return torch.tensor(targets, dtype=torch.long).to(self.device), torch.tensor(lengths, dtype=torch.long).to(self.device)

    def decode_predictions(self, log_probs: torch.Tensor):
        log_probs_np = log_probs.cpu().detach().numpy()
        predictions = []

        if self.model.decoder_type == DecoderType.WordBeamSearch:
            char_map = {c: i for i, c in enumerate(self.char_list)}

            for probs in log_probs_np:
                wbs_prediction = WordBeamSearch(
                    50,
                    "Words",
                    0.0,
                    self.model.corpus,
                    "".join(self.char_list),
                    self.model.word_chars,
                )
                predictions.append(wbs_prediction)

        elif self.model.decoder_type == DecoderType.BestPath:
            pred_indices = torch.argmax(log_probs, dim=2)
            for pred in pred_indices:
                decoded = []
                prev_char = None
                for index in pred:
                    if index != prev_char and index != len(self.char_list):
                        decoded.append(self.char_list[index])
                    prev_char = index
                predictions.append("".join(decoded))

        return predictions
