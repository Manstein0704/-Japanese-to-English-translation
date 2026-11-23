"""Training and checkpointing (to implement)."""
import torch
import torch.nn as nn
from translator.model import Transformer
from tokenizers import Tokenizer
from translator.data import (load_jesc_datasets, tokenize_dataset, get_batch)
import random
import numpy as np
from pathlib import Path
import csv

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)


BATCH_SIZE = 8
LEARNING_RATE = 3e-4
MAX_STEPS = 1000000
SEED=2026

def main():
    set_seed(seed=SEED)

    #setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load Dataset and Tokenizer
    train_ds, eval_ds, test_ds = load_jesc_datasets()
    tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")

    #tokenize dataset
    train_data = tokenize_dataset(train_ds, tokenizer)
    eval_data = tokenize_dataset(eval_ds, tokenizer)
    test_data = tokenize_dataset(test_ds, tokenizer)

    #model initialization
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    model = Transformer(vocab_size=vocab_size, pad_id=pad_id).to(device)

    #define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)


    log_path = Path("outputs/train_loss.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    total_loss = 0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "TrainLoss"])

    for step in range(MAX_STEPS):

        #Get Batch
        src, decoder_input, labels = get_batch(train_data, batch_size=BATCH_SIZE, pad_id=pad_id)
        src = src.to(device)
        decoder_input = decoder_input.to(device)
        labels = labels.to(device)

        #Forward
        logits = model(src=src, decoder_input=decoder_input)

        #get loss
        loss = criterion(logits.reshape(-1, vocab_size),
                        labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / 100
            print(
                f"Step {step+1}/{MAX_STEPS} | "
                f"Train Loss: {avg_loss:.4f}"
            )

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step+1, avg_loss])
            total_loss = 0.0



    save_path = Path(f"artifacts/checkpoints/minigpt_step{MAX_STEPS:06d}.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {}
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": MAX_STEPS,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "seed": SEED,
        },
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()