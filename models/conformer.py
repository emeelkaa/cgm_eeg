# EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization
# https://github.com/eeyhsong/EEG-Conformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, n_channels):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 25), (1, 1)),
            nn.Conv2d(emb_size, emb_size, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),     # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, (1, 1), stride=(1, 1)), 
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    
    def forward(self, x):
        x = self.shallownet(x)
        out = self.projection(x)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, heads, drop_p):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.d_k = emb_size // heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(drop_p)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)

        energy = torch.einsum('b h q d, b h k d -> b h q k', queries, keys) 
        scaling = self.d_k ** (1 / 2)

        att = F.softmax(energy / scaling, dim=-1)
        att = self.dropout(att)
        out = torch.einsum('b h a l, b h l v -> b h a v ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, emb_size, heads, drop_p=0.5, forward_expansion=4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, forward_expansion, drop_p),
                nn.Dropout(drop_p)
            )
            ))

class Conformer(nn.Module):
    def __init__(self, n_channels, n_classes, emb_size=40, depth=6, heads=10, drop_p=0.5, forward_expansion=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.encoder = nn.ModuleList([
            TransformerEncoder(emb_size, heads, drop_p, forward_expansion)
            for _ in range(depth)
        ])

        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
    
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.patch_embedding(x)
        for transformer in self.encoder:
            x = transformer(x)
        out = self.cls_head(x)
        return out
    

if __name__ == '__main__':
    import time
    model = Conformer(emb_size=64, depth=4, heads=4, n_channels=16, n_classes=1).to('cuda')
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(1, 16, 2000).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")
    