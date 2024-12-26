"""
From https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import math


# Hyperparameters


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    n_layer: int = 8


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dro
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((config.block_size, config.block_size))),
        ).view(1, 1, config.block_size, config.block_size)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(2, 1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(2, 1)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(2, 1)
        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_embd))
        wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm()
        self.ff = FeedForward(config)
        self.ln2 = nn.LayerNorm()

    def forward(self, x):
        x = x + self.attn(x)
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.attn = nn.Sequential(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.ln = nn.LayerNorm()
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        embeds = self.embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long)
        pos_embeds = self.pos_embedding(pos)
        out = self.dropout(embeds + pos_embeds)
        out = self.attn(out)
        out = self.ln(out)

        if targets:
            out = self.lm_head(out)
            loss = F.cross_entropy(out, targets)
        else:
            out = self.lm_head(out[:, [-1], :])
            loss = None

        return out, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, _ = self.forward(idx)  # B, 1, N
            preds = torch.argmax(logits, dim=-1)  # or sample from multinomial
            # Crop if growing long
            idx = idx[:, -self.block_size :]
            idx = torch.cat((idx, preds), dim=-1)
        return idx
