# Japanese-to-English Translation with Transformer

A Japanese-to-English neural machine translation project built from scratch with PyTorch. It uses an encoder–decoder Transformer trained on the [JESC dataset](https://huggingface.co/datasets/nntsuzu/JESC).

The goal is to understand and implement the full machine translation pipeline, from tokenization and model training to evaluation and autoregressive inference.

## Features

- Byte-level BPE tokenizer trained on Japanese and English text
- Encoder–decoder Transformer with self-attention and cross-attention
- Teacher forcing during training and greedy decoding for inference
- Training and validation loss logging to CSV
- Dataset caching and model checkpoints

## Model Configuration

| Setting | Value |
| --- | --- |
| Embedding dimension | 384 |
| Attention heads | 8 |
| Encoder / decoder layers | 4 / 4 |
| Feed-forward dimension | 1,536 |
| Dropout | 0.1 |
| Maximum sequence length | 128 tokens |
| Optimizer | AdamW |

These are the settings currently used in `train.py`; they can be changed for experiments.

## Getting Started

Run all commands from the repository root.

```bash
git clone https://github.com/Manstein0704/-Japanese-to-English-translation.git
cd -- -Japanese-to-English-translation
python -m venv venv
source venv/bin/activate
pip install torch datasets tokenizers numpy
```

### 1. Prepare the Tokenizer

```bash
PYTHONPATH=src python -m translator.tokenizer
```

The tokenizer is saved to `artifacts/tokenizer.json`. The dataset is downloaded or loaded from a local cache during preparation.

### 2. Train the Model

```bash
PYTHONPATH=src python -m translator.train
```

The training script records training and validation losses in `outputs/train_loss.csv` and saves a checkpoint after training finishes. Training settings, including the number of steps, are configured in `src/translator/train.py`.

### 3. Translate Text

After training, set `CHECKPOINT_PATH` in `src/translator/translate.py` to an existing checkpoint and ensure the model configuration matches the checkpoint.

```bash
PYTHONPATH=src python -m translator.translate
```

Enter Japanese text at the prompt. Type `quit` to exit.

## Project Structure

```text
src/translator/
├── data.py        # JESC loading, caching, and batches
├── tokenizer.py   # BPE tokenizer training
├── model.py       # Encoder–decoder Transformer
├── train.py       # Training and checkpointing
├── evaluate.py    # Validation loss
└── translate.py   # Greedy autoregressive decoding
```

## Status

This is an experimental learning project. Translation quality is still being evaluated and improved; no production-level accuracy is claimed.

## Dataset

[JESC (Japanese–English Subtitle Corpus)](https://huggingface.co/datasets/nntsuzu/JESC), a parallel corpus derived from subtitles.
