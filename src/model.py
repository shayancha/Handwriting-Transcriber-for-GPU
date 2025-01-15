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

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

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
        best_loss = float("inf")

        for epoch_num, epoch in enumerate(range(num_epochs), start=1):
            self.console.print(f"\n[bold green]Epoch {epoch_num}/{num_epochs}[/bold green]")
            self.model.train()
            epoch_losses = []

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

                    epoch_losses.append(loss.item())

                    tbar.update(1)
                    tbar.set_postfix({"loss": loss.item(), "batch_time": f"{time.time()-start_time:.2f}s"})

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.console.print(f"[bold blue]Training Loss:[/bold blue] {avg_loss:.4f}")

            val_loss = self.validate(val_loader)
            print(f"[bold yellow]Validation loss:[/bold yellow] {val_loss:.4f}")

            model_path = os.path.join(save_dir, f"model_epoch_{epoch_num}.pth")
            torch.save(self.model.state_dict(), model_path)
            self.console.print(f"[bold green]Model saved at {model_path}[/bold green]")

            if val_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                self.console.print(f"[bold green]Best model saved at {best_model_path} with loss: {best_loss:.4f}[/bold green]")

    def validate(self, val_loader: TorchDataLoader) -> float:
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            with tqdm(val_loader, desc="Validating", leave=False) as tbar:
                for batch in val_loader:
                    images = batch["images"].to(self.device)
                    ground_truths = batch["ground_truths"]

                    logits = self.model(images)
                    log_probs = F.log_softmax(logits, dim=2)

                    targets, targtet_lengths = self.encode_targets(ground_truths)
                    input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)

                    loss = self.ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
                    val_losses.append(loss.item())

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
        pred_indices = torch.argmax(log_probs, dim=2)
        predictions = []

        for pred in pred_indices:
            decoded = []
            prev_char = None
            for index in pred:
                if index != prev_char and index != len(self.char_list):
                    decoded.append(self.char_list[index])
                prev_char = index
            predictions.append("".join(decoded))

        return predictions
