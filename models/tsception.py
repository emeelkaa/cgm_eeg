# TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition
# https://github.com/yi-ding-cs/TSception

import torch 
import torch.nn as nn

class TSception(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, n_channels, sfreq, n_classes, t_emb=32, s_emb=64, hidden=64, pool=8, drop_p=0.3):
        super().__init__()
        self.inception_windows = [0.5, 0.25, 0.125]

        self.tception = nn.ModuleList([
            self.conv_block(1, t_emb, (1, int(w * sfreq)), 1, pool) 
            for w in self.inception_windows 
        ])
        self.t_norm = nn.BatchNorm2d(t_emb)

        self.sception = nn.ModuleList([
            self.conv_block(t_emb, s_emb, (n_channels, 1), 1, int(pool * 0.25)),
            self.conv_block(t_emb, s_emb, (int(n_channels * 0.5), 1), (int(n_channels * 0.5), 1), int(pool * 0.25)),
        ])
        self.s_norm = nn.BatchNorm2d(s_emb)

        self.fusion_layer = self.conv_block(s_emb, s_emb, (3, 1), 1, 4)
        self.fusion_norm = nn.BatchNorm2d(s_emb)

        self.cls_head = nn.Sequential(
            nn.Linear(s_emb, hidden),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x_t = [block(x) for block in self.tception]
        out = torch.cat(x_t, dim=-1)

        x_s = [block(out) for block in self.sception]
        out = torch.cat(x_s, dim=2)
        out = self.s_norm(out)

        out = self.fusion_layer(out)
        out = self.fusion_norm(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.cls_head(out)
        return out

if __name__ == "__main__":
    import time
    model = TSception(n_channels=16, sfreq=200, n_classes=1).to('cuda')
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(1, 16, 2500).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")
    