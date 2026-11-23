"""Validation and translation metrics (to implement)."""
import torch
import torch.nn as nn
from translator.data import get_batch

@torch.inference_mode()
def evaluate(model, data, criterion, device, vocab_size, pad_id, batch_size=8, eval_iters=20):
    model.eval()
    total_loss = 0.0

    for _ in range(eval_iters):
        src, decoder_input, labels = get_batch(data,
                                               batch_size=batch_size,
                                               pad_id=pad_id)
        src = src.to(device)
        decoder_input = decoder_input.to(device)
        labels = labels.to(device)

        logits = model(src, decoder_input)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1)
        )

        total_loss += loss.item()

    avg_loss = total_loss / eval_iters

    model.train()

    return avg_loss



