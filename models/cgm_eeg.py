import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from einops.layers.torch import Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, num_channels, n_fft=200, hop_length=100):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.segment_embedding = nn.Linear(self.n_fft // 2 + 1, emb_size)
        self.channel_tokens = nn.Embedding(num_channels, emb_size)
        self.index = nn.Parameter(torch.LongTensor(range(num_channels)), requires_grad=False)

        self.register_buffer("window", torch.hann_window(self.n_fft))

    def stft(self, sample):
        sample_flat = rearrange(sample, 'b c t -> (b c) t')
        spectral = torch.stft(
            input=sample_flat,
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        spec = self.stft(x)
        spec = rearrange(spec, '(b c) f t -> b c t f', b=x.shape[0])
        emb = self.segment_embedding(spec)
        channel_token_emb = self.channel_tokens(self.index)
        emb = emb + channel_token_emb.unsqueeze(0).unsqueeze(2)  
        return emb
        
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class BiMambaBlock(nn.Module):
    def __init__(self, emb_size, d_state=16, d_conv=4, drop_p=0.5):
        super().__init__()

        self.mamba_fwd = Mamba(d_model=emb_size, d_state=d_state, d_conv=d_conv)
        self.ln1 = nn.LayerNorm(emb_size)
        self.mamba_rev = Mamba(d_model=emb_size, d_state=d_state, d_conv=d_conv)
        self.ln1_rev = nn.LayerNorm(emb_size)

        self.ffn = FeedForwardBlock(emb_size, 4, drop_p)
        self.ln2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(drop_p)
    
    def forward_branch(self, x, mamba, ln1, flip_time=False):
        if flip_time: 
            x_in = torch.flip(x, dims=[1]) 
        else:
            x_in = x
        y = self.dropout(mamba(x_in))
        if flip_time:
            y = torch.flip(y, dims=[1])
        out = ln1(x + y)
        return out

    def forward(self, x):
        out_fwd = self.forward_branch(x, self.mamba_fwd, self.ln1)
        out_rev = self.forward_branch(x, self.mamba_rev, self.ln1_rev, flip_time=True)
        out = 0.5 * (out_fwd + out_rev)
        out = self.ln2(out + self.dropout(self.ffn(out)))
        return out

class CrossGatedModule(nn.Module):
    def __init__(self, emb_size, n_channels, drop_p=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.gate_t2s = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, 1),
        )
        self.gate_s2t = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, 1),
        )

        self.proj_t = nn.Linear(emb_size, emb_size)
        self.proj_s = nn.Linear(emb_size, emb_size)

        self.ln_t = nn.LayerNorm(emb_size)
        self.ln_s = nn.LayerNorm(emb_size)
        self.dropout_t = nn.Dropout(drop_p)
        self.dropout_s = nn.Dropout(drop_p)
    
    def forward(self, f_t, f_s):
        C = self.n_channels
        T = f_t.shape[1] // C

        f_t = rearrange(f_t, 'b (c t) d -> b c t d', c=C, t=T).contiguous()
        f_s = rearrange(f_s, 'b (c t) d -> b c t d', c=C, t=T).contiguous()

        g_s2t = torch.sigmoid(self.gate_s2t(torch.cat([f_t.mean(dim=1), f_s.mean(dim=1)], dim=-1)))
        f_t_out = self.ln_t(self.dropout_t(f_t + (g_s2t.unsqueeze(1) * self.proj_s(f_s))))

        g_t2s = torch.sigmoid(self.gate_t2s(torch.cat([f_t.mean(dim=2), f_s.mean(dim=2)], dim=-1)))
        f_s_out = self.ln_s(self.dropout_s(f_s + (g_t2s.unsqueeze(2) * self.proj_t(f_t_out))))

        return f_t_out, f_s_out
    
class CGM(nn.Module):
    def __init__(self, emb_size=64, depth=2, n_channels=16, n_classes=1, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.patch_embedding = PatchEmbedding(emb_size, n_channels, **kwargs)
        self.mamba_t = nn.ModuleList(
            [BiMambaBlock(emb_size=emb_size) for _ in range(depth)]
        )
        self.mamba_s = nn.ModuleList(
            [BiMambaBlock(emb_size=emb_size) for _ in range(depth)]
        )
        self.cgm = nn.ModuleList([CrossGatedModule(emb_size, n_channels) for _ in range(depth)])

        self.residual_gates = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(depth)])

        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        f_e = self.patch_embedding(x)

        for i, (layer_t, layer_s) in enumerate(zip(self.mamba_t, self.mamba_s)):
            f_e_prev = f_e

            f_s = rearrange(f_e, "b c t d -> b (t c) d")
            f_s = layer_s(f_s)
            f_s = rearrange(f_s, "b (t c) d -> b (c t) d", c=self.n_channels)
            
            f_t = rearrange(f_e, "b c t d -> b (c t) d")
            f_t = layer_t(f_t)
            
            f_t, f_s = self.cgm[i](f_t, f_s)

            f_e_new = 0.5 * (f_t + f_s)

            gamma = torch.sigmoid(self.residual_gates[i])
            f_e = gamma * f_e_new + (1 - gamma) * f_e_prev

        out = self.cls_head(rearrange(f_e, "b c t d -> b (c t) d"))
        return out


if __name__ == "__main__":
    import time
    n_channels = 16
    model = CGM(emb_size=64, depth=2, n_channels=n_channels, n_classes=1, n_fft=200, hop_length=100).to('cuda')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    x = torch.randn(1, n_channels, 2000).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(out.shape)