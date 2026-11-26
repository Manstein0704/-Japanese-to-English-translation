"""Autoregressive translation inference (to implement)."""

import torch
from pathlib import Path
from tokenizers import Tokenizer
from translator.model import Transformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "artifacts/tokenizer.json"
CHECKPOINT_PATH = "artifacts/checkpoints/model_step100000.pt"

def load_model():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id(("<pad>"))

    model = Transformer(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=128,
        pad_id=pad_id
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH,
                            map_location=device,
                            weights_only=True)


    print("Checkpoint:", CHECKPOINT_PATH)
    print("Training Step:", checkpoint["step"])
    print("Training Config:", checkpoint["training_config"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


@torch.inference_mode()
def translate(model, tokenizer:Tokenizer, text, max_new_tokens=1000):
    src_ids = tokenizer.encode(text).ids
    if len(src_ids) > model.max_len:
        raise ValueError("Input Sequence exceeds max_len")

    src = torch.tensor([src_ids],
                       dtype=torch.long,
                       device=device)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    decoder_input = torch.tensor([[bos_id]],
                                 dtype=torch.long,
                                 device=device)

    for _ in range(min(max_new_tokens, model.max_len - 1)):
        logits = model(src, decoder_input)

        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )

        decoder_input = torch.cat(
            [decoder_input, next_token],
            dim=1
        )

        if next_token.item() == eos_id:
            break

    result = tokenizer.decode(
        decoder_input[0].tolist(),
        skip_special_tokens=True
    )

    return result

def main():
    model, tokenizer = load_model()

    while True:
        text = input("Japanese (quit to exiit)")

        if text.lower() == "quit":
            break

        result = translate(model, tokenizer, text)
        print(f"English:{result}\n")


if __name__ == "__main__":
    main()
    




    

    

