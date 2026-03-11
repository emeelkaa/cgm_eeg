# Development of Expert-Level Classification of Seizures and Rhythmic and 
# Periodic Patterns During EEG Interpretation
# Refer to BIOT repository:
# https://github.com/ycq091044/BIOT

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, expansion, bn_size, drop_p, conv_bias, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm1 = nn.BatchNorm1d(in_channels)
            self.norm2 = nn.BatchNorm1d(bn_size * expansion)
        
        self.net1 = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channels, (bn_size * expansion), 1, 1, bias=conv_bias)
        )
        self.net2 = nn.Sequential(
            nn.ELU(),
            nn.Conv1d((bn_size * expansion), expansion, 3, 1, 1, bias=conv_bias)
        )
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x):
        if self.batch_norm:
            x_in = self.norm1(x)
        else:
            x_in = x 

        out = self.net1(x_in)
        if self.batch_norm:
            out = self.norm2(out)
        
        out = self.net2(out)
        out = self.dropout(out)
        return torch.cat([x, out], 1)
    
class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_channels, expansion, bn_size, drop_p, conv_bias, batch_norm):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + idx_layer * expansion, expansion, bn_size, drop_p, conv_bias, batch_norm) 
            for idx_layer in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv_bias, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm = nn.BatchNorm1d(in_channels)
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channels, out_channels, 1, 1, bias=conv_bias),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        if self.batch_norm:
            x = self.norm(x)
        out = self.net(x)
        return out

class SPaRCNet(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_classes, block_layers=4, expansion=16, bn_size=16, drop_p=0.5, conv_bias=True, batch_norm=True):
        super().__init__()
        out_channels = 2 ** (math.floor(np.log2(n_channels)) + 1)
        
        self.net0 = nn.Sequential(
            nn.Conv1d(n_channels, out_channels, 7, 2, 3, bias=conv_bias),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        in_channels = out_channels

        for n_layer in np.arange(math.floor(np.log2(n_timepoints // 4))):
            block = DenseBlock(
                n_layers=block_layers,
                in_channels=in_channels,
                expansion=expansion,
                bn_size=bn_size,
                drop_p=drop_p,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.dense_blocks.append(block)
            in_channels = in_channels + block_layers * expansion

            transition = TransitionLayer(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.transitions.append(transition)
            in_channels = in_channels // 2
        
        self.cls_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_channels, n_classes)
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net0(x)
        
        for block, transition in zip(self.dense_blocks, self.transitions):
            x = block(x)
            x = transition(x)
        
        out = self.cls_head(x.squeeze(-1))
        return out

if __name__ == "__main__":
    import time
    model = SPaRCNet(n_channels=16, n_timepoints=2000, n_classes=1).to('cuda')
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(1, 16, 2000).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")