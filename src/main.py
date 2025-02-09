import torch
import argparse
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel
from dataset import HandwritingDataset
from model import load_trocr_model
from preprocess import TrOCRImageProcessor
from PIL import Image
import time


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is not None:
            inputs.pop("num_items_in_batch", None)

        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def train(img_dir, metadata_path, output_dir, batch_size=16, epochs=10, learning_rate=5e-5):
    print("Initializing TrOCR Model & Processor...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    # for faster inference and training use below
    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")

    model = load_trocr_model()
    image_processor = TrOCRImageProcessor()

    model.config.decoder_start_token_id = model.config.pad_token_id

    print("Loading Dataset...")
    full_dataset = HandwritingDataset(img_dir, metadata_path, processor.tokenizer, image_processor, augment=True)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.augment = True
    val_dataset.dataset.augment = False

    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()

    print(f"Using device: {device} (FP16: {'Enabled' if use_fp16 else 'Disabled'})")

    print("Setting Training Arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=100,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        num_train_epochs=epochs,
        warmup_steps=1000,
        fp16=use_fp16,
        push_to_hub=False
    )

    print("Starting Training...")
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    print("✅ Training Complete. Model saved to:", output_dir)


def infer(model_checkpoint_dir, img_path):
    start_time = time.time()
    print("🚀 Loading Fine-Tuned TrOCR Model for Inference...")

    model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint_dir)
    # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    # for faster inference and training use below
    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")


    print("📸 Preprocessing Image...")
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    print("🔍 Running Inference...")
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"📝 Predicted Text: {predicted_text}")
    print(f"Time it took:{time.time() - start_time}")
    return predicted_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR or Run Inference")

    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True, help="Choose 'train' or 'infer'")
    parser.add_argument("--img_dir", type=str, default=None, help="Path to image dataset (Required for training)")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata file (Required for training)")
    parser.add_argument("--output_dir", type=str, default="./trocr_finetuned_models", help="Where to save trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--model_path", type=str, default="./trocr_finetuned_models", help="Path to trained model (Required for inference)")
    parser.add_argument("--img_path", type=str, default=None, help="Path to single image for inference")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.img_dir, args.metadata, args.output_dir, args.batch_size, args.epochs, args.lr)

    elif args.mode == "infer":
        infer(args.model_path, args.img_path)
