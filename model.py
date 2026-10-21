import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.dropout_prob = dropout
        
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv_proj(x).split(C, -1)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_prob if self.training else 0.00,
            is_causal=True
        ).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

class Transformer(nn.Module):
    def __init__(self, num_embeddings, block_size, embedding_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.block_size = block_size

        self.token_emb = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_emb = nn.Embedding(block_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        layers = [Block(embedding_dim, num_heads, dropout) for _ in range(num_layers)]
        self.net = nn.Sequential(
            *layers,
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_embeddings, bias=False)
        )
        self.net[-1].weight = self.token_emb.weight
        self.apply(self._init_weights)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size

        pos = self.pos_emb(torch.arange(0, T, device=idx.device))
        token = self.token_emb(idx)
        x = self.dropout(pos + token)
        return self.net(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        