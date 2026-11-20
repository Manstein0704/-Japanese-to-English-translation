"""Training and checkpointing (to implement)."""
import torch
import torch.nn as nn
from translator.model import Transformer
from tokenizers import Tokenizer
from translator.data import (load_jesc_datasets, tokenize_dataset, get_batch)
import random
import numpy as np
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)


BATCH_SIZE = 8
LEARNING_RATE = 3e-4
MAX_STEPS = 10000
SEED=2026

def main(num_step=1000):
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

    model.train()
    for step in range(MAX_STEPS):

        #Get Batch
        src, decoder_input, labels = get_batch(train_data)
        src = src.to(device)
        decoder_input = decoder_input.to(device)
        labes = labels.to(device)

        #Forward
        logits = model.forward(src=src, decoder_input=decoder_input)

        #get loss
        loss = criterion(logits.reshape(-1, vocab_size),
                         labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(
                f"Step {step+1}/{MAX_STEPS} | "
                f"Batch Loss: {loss.item():.4f}"
            )





