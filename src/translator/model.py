"""Encoder-decoder Transformer architecture (to implement)."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, max_len=128, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        #Token Embedding
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        #Positional_Embedding
        self.src_pos_embedding = nn.Embedding(max_len, d_model)
        self.tgt_pos_embedding = nn.Embedding(max_len, d_model)
        
        #Encoder-Decoder Transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True,
                                          norm_first=True)

        #Output Linear Layer
        self.linear = nn.Linear(d_model, vocab_size)







    def forward(self, src, decoder_input):
        B, S = src.shape
        _, T = decoder_input.shape

        if S > self.max_len or T>self.max_len:
            raise ValueError(f"Sequence length exceeds max_len={self.max_len}")

        src_positions = torch.arange(S, device=src.device)
        tgt_positions = torch.arange(T, device=decoder_input.device)

        src_pos = self.src_pos_embedding(src_positions)
        tgt_pos = self.tgt_pos_embedding(tgt_positions)

        src_emb = self.src_embedding(src) + src_pos
        tgt_emb = self.tgt_embedding(decoder_input) + tgt_pos

        attention_mask = nn.Transformer.generate_square_subsequent_mask(
            T,
            device=decoder_input.device
        )
        src_padding_mask = (src==self.pad_id)
        tgt_padding_mask = (decoder_input==self.pad_id)

        hidden = self.transformer(src=src_emb,
                                  tgt=tgt_emb,
                                  tgt_mask=attention_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask)

        logits = self.linear(hidden)

        return logits

    
if __name__ == "__main__":
    model = Transformer(vocab_size=24000)
    src = torch.randint(4, 24000, (2, 20))
    decoder_input = torch.randint(4, 24000, (2, 15))

    with torch.no_grad():
        logits = model(src, decoder_input)
    print("Logits shape:", logits.shape)
        
        









