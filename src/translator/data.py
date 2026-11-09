"""Download and split the JESC Japanese-English parallel corpus.

Run:
    PYTHONPATH=src python -m translator.data

Output:
    data/processed/{train,validation,test}/  (Hugging Face Dataset format)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

DATASET_NAME = "nntsuzu/JESC"
DEFAULT_OUTPUT_DIR = Path("data/processed")
SEED = 2026
EVAL_SIZE = 2_000


def valid_pair(example: dict) -> bool:
    """Reject rows that do not contain both languages."""
    pair = example["translation"]
    return bool(pair["ja"] and pair["ja"].strip() and pair["en"] and pair["en"].strip())


def prepare_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = SEED,
) -> DatasetDict:
    """Fetch JESC, make deterministic splits, and store them locally."""
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; choose another output path "
            "or load the existing dataset."
        )

    print(f"Downloading {DATASET_NAME}...")
    full = load_dataset(DATASET_NAME, split="train")
    full = full.filter(valid_pair, desc="Removing empty sentence pairs")

    if len(full) <= EVAL_SIZE * 2:
        raise ValueError("Not enough sentence pairs for train/validation/test splits")

    # Hold out 4,000 examples; divide equally into validation and test.
    split = full.train_test_split(test_size=EVAL_SIZE * 2, seed=seed)
    held_out = split["test"].train_test_split(test_size=EVAL_SIZE, seed=seed)
    dataset = DatasetDict(
        {
            "train": split["train"],
            "validation": held_out["train"],
            "test": held_out["test"],
        }
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local JESC splits")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.output_dir.exists():
        print(f"Loading existing splits from {args.output_dir}")
        dataset = load_from_disk(str(args.output_dir))
    else:
        dataset = prepare_dataset(args.output_dir, args.seed)

    print("\nDataset splits:")
    for name, split in dataset.items():
        print(f"  {name}: {len(split):,} pairs")

    print("\nFirst 3 training examples:")
    for i in range(min(3, len(dataset["train"]))):
        pair = dataset["train"][i]["translation"]
        print(f"[{i + 1}] JA: {pair['ja']}")
        print(f"    EN: {pair['en']}")


if __name__ == "__main__":
    main()
