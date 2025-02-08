from transformers import VisionEncoderDecoderModel


def load_trocr_model():
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id

    if model.config.pad_token_id is None:
        model.config.pad_token_id = 1

    if not hasattr(model.decoder.config, "vocab_size"):
        print("⚠️ WARNING: `vocab_size` not found in `decoder.config`, setting it manually.")
        model.decoder.config.vocab_size = 50265

    model.config.max_length = 512

    return model
