import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from einops.layers.torch import Reduce


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_channels: int,
            sfreq: int
    ):
        super().__init__()
        self.n_fft = sfreq
        self.hop_length = sfreq // 2

        self.segment_embedding = nn.Linear(self.n_fft // 2 + 1, d_model)
        self.channel_tokens = nn.Embedding(num_channels, d_model)
        self.index = nn.Parameter(torch.LongTensor(range(num_channels)), requires_grad=False)

        self.register_buffer("window", torch.hann_window(self.n_fft))

    def stft(self, sample):
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.segment_embedding(
                rearrange(channel_spec_emb, "b d t -> b t d")
            )
            B, T, _ = channel_spec_emb.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, T, 1)
            )
            channel_emb = channel_spec_emb + channel_token_emb
            emb_seq.append(channel_emb)
        emb = torch.stack(emb_seq, dim=1)
        return emb
        

class BiMambaBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            drop_p: float = 0.3
    ):
        super().__init__()

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.mamba_rev = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln1_rev = nn.LayerNorm(d_model)
        self.ln2_rev = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(drop_p),
        )

        self.ffn_rev = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(drop_p),
        )

        self.dropout = nn.Dropout(drop_p)
    
    def forward_branch(self, x, mamba, ln1, ln2, ffn, flip_time=False):
        if flip_time: 
            x_in = torch.flip(x, dims=[1]) 
        else:
            x_in = x
        
        y = mamba(x_in)
        y = self.dropout(y)

        if flip_time:
            y = torch.flip(y, dims=[1])

        y = ln1(x + y)
        y2 = ffn(y)
        y2 = self.dropout(y2)
        y = ln2(y + y2)
        return y

    def forward(self, x):
        out_fwd = self.forward_branch(
            x, self.mamba_fwd, self.ln1, self.ln2, self.ffn, flip_time=False
        )
        out_rev = self.forward_branch(
            x, self.mamba_rev, self.ln1_rev, self.ln2_rev, self.ffn_rev, flip_time=True
        )
        out = 0.5 * (out_fwd + out_rev)
        return out


class CrossGatedModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_channels: int,
            drop_p: float = 0.1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.gate_t2s = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.gate_s2t = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        self.proj_t = nn.Linear(d_model, d_model)
        self.proj_s = nn.Linear(d_model, d_model)

        self.ln_t = nn.LayerNorm(d_model)
        self.ln_s = nn.LayerNorm(d_model)
        self.dropout_t = nn.Dropout(drop_p)
        self.dropout_s = nn.Dropout(drop_p)
    
    def forward(self, f_t_flat, f_s_flat):
        B, NT, D = f_t_flat.shape
        C = self.num_channels
        T = NT // C

        f_t = rearrange(f_t_flat, 'b (c t) d -> b c t d', c=C, t=T).contiguous()
        f_s = rearrange(f_s_flat, 'b (c t) d -> b c t d', c=C, t=T).contiguous()

        t_c = f_t.mean(dim=1)
        s_c = f_s.mean(dim=1)
        g_s2t = torch.sigmoid(self.gate_s2t(torch.cat([t_c, s_c], dim=-1)))

        t_t = f_t.mean(dim=2)
        s_t = f_s.mean(dim=2)
        g_t2s = torch.sigmoid(self.gate_t2s(torch.cat([t_t, s_t], dim=-1)))

        f_t_out = f_t + (g_s2t.unsqueeze(1) * self.proj_s(f_s))
        f_s_out = f_s + (g_t2s.unsqueeze(2) * self.proj_t(f_t))

        f_t_out = self.ln_t(f_t_out)
        f_s_out = self.ln_s(f_s_out)
        f_t_out = self.dropout_t(f_t_out)
        f_s_out = self.dropout_s(f_s_out)

        f_t_out = rearrange(f_t_out, 'b c t d -> b (c t) d')
        f_s_out = rearrange(f_s_out, 'b c t d -> b (c t) d')
        return f_t_out, f_s_out
    

class ClassificationHead(nn.Sequential):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class CGM(nn.Module):
    def __init__(
            self,
            d_model: int = 64,
            depth: int = 2,
            num_channels: int = 16,
            sfreq: int = 256,
            num_classes: int = 1
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(d_model, num_channels, sfreq)
        self.mamba_t = nn.ModuleList(
            [BiMambaBlock(d_model=d_model) for _ in range(depth)]
        )
        self.mamba_s = nn.ModuleList(
            [BiMambaBlock(d_model=d_model) for _ in range(depth)]
        )
        self.ln_t = nn.LayerNorm(d_model)
        self.ln_s = nn.LayerNorm(d_model)
        self.cgm = CrossGatedModule(d_model, num_channels)
        self.cls_head = ClassificationHead(d_model, num_classes)

    def forward(self, x):
        f_e = self.patch_embedding(x)

        for layer_t, layer_s in zip(self.mamba_t, self.mamba_s):
            f_s = rearrange(f_e, "b c t d -> (b t) c d")
            f_s = layer_s(f_s)
            f_s = rearrange(f_s, "(b t) c d -> b (c t) d", b=x.shape[0])
            f_s = self.ln_s(f_s)
            
            f_t = rearrange(f_e, "b c t d -> (b c) t d")
            f_t = layer_t(f_t)
            f_t = rearrange(f_t, "(b c) t d -> b (c t) d", b=x.shape[0])
            f_t = self.ln_t(f_t)
            
            f_t, f_s = self.cgm(f_t, f_s)
        
        out = 0.5 * (f_t + f_s)
        out = self.cls_head(out)
        return out


if __name__ == "__main__":
    import time
    model = CGM().to('cuda')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    x = torch.randn(1, 16, 2560).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(out.shape)