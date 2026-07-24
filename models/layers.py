import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------
# LayerNorm for channel‑first tensors (B, C, H, W)
# -------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x    = (x - mean) / (var + self.eps).sqrt()
        return self.weight[:, None, None] * x + self.bias[:, None, None]

# -------------------------------------------------------------------
# Traditional conv2D block used in v2 decoders
# -------------------------------------------------------------------
class conv2D_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# -------------------------------------------------------------------
# GatedCNN block – core of MambaOut
# -------------------------------------------------------------------
class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expand_ratio=8/3, kernel_size=7):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        hidden = int(dim * expand_ratio)
        hidden = hidden + (hidden % 2)          # even for chunk
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.dw  = nn.Conv2d(hidden, hidden, kernel_size,
                             padding=kernel_size//2, groups=hidden, bias=False)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        v, g = self.fc1(x).chunk(2, dim=1)
        v = self.dw(v)
        x = v * F.gelu(g)
        x = self.fc2(x)
        return x + shortcut