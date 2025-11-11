import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, drop_p: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(drop_p)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, d_model: int, n_channels: int, sfreq: int):
        super().__init__()
        self.n_fft = sfreq
        self.hop_length = sfreq // 2
        self.projection = nn.Linear(self.n_fft // 2 + 1, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.channel_tokens = nn.Embedding(n_channels, d_model)
        self.index = nn.Parameter(torch.LongTensor(range(n_channels)), requires_grad=False)
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            window=self.window,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.projection(
                rearrange(channel_spec_emb, "b d t -> b t d")
            )
            B, T, _ = channel_spec_emb.shape
            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, T, 1)
            )
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)
            emb_seq.append(channel_emb)
        emb = torch.cat(emb_seq, dim=1)
        return emb


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()

        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.clshead(x)
        return out
    

class BIOT(nn.Module):
    def __init__(
            self,
            d_model: int = 64,
            num_heads: int = 4,
            depth: int = 4,
            num_channels: int = 16,
            sfreq: int = 256,
            num_classes: int = 1
    ):
        super().__init__()
        self.patch_embedding = PatchFrequencyEmbedding(d_model, num_channels, sfreq)
        self.transformer = LinearAttentionTransformer(
            dim=d_model, 
            heads=num_heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.classifier = ClassificationHead(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x).mean(dim=1)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    import time
    model = BIOT().to('cuda')
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(1, 16, 2560).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")
