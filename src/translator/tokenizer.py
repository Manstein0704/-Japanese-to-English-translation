from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from translator.data import load_jesc_datasets




def train_tokenizer(train_ds, output_path:Path=Path("artifacts/tokenizer.json")):
    if output_path.exists():
        print("Loading existing tokenizer...")
        tokenizer = Tokenizer.from_file(str(output_path))
        return tokenizer
    
    else:
        print("Training new tokenizer...")

    VOCAB_SIZE=24000
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    tokenizer = Tokenizer(models.BPE(unk_token = "<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    def text_iterator():
        for example in train_ds:
            jap = example["translation"]["ja"]
            eng = example["translation"]["en"]
            yield jap
            yield eng

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer,
        length = len(train_ds)*2
    )

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id)
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    return tokenizer




if __name__ == "__main__":
   train_ds, _, _ = load_jesc_datasets()
   tokenizer_train_ds = train_ds.shuffle(seed=2026).select(range(1000000))
   tokenizer = train_tokenizer(train_ds=tokenizer_train_ds)

   sample_text = ["今日はとてもいい日ですね",
                  "Today is a beautiful day."]

   for text in sample_text:
       encoded = tokenizer.encode(text)
       print(f"\nOriginal:{text}")
       print(f"Tokens:{encoded.tokens}")
       print(f"IDs:{encoded.ids}")
       print(f"Decoded:{tokenizer.decode(encoded.ids)}")


