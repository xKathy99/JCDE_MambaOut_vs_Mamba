import torch
import torch.nn as nn
from .layers import LayerNorm2d, conv2D_block
from mamba_ssm import Mamba   # requires pip install mamba-ssm

class MambaBlock(nn.Module):
    """2D wrapper for Mamba with 4‑directional scanning."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def _scan(self, tokens):
        return self.mamba(self.norm(tokens))

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        tokens = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        out  = self._scan(tokens)
        out += self._scan(tokens.flip(1)).flip(1)
        out += self._scan(tokens.reshape(B, H, W, C).permute(0,2,1,3).reshape(B, H*W, C))
        out += self._scan(tokens.reshape(B, H, W, C).permute(0,2,1,3).reshape(B, H*W, C).flip(1)).flip(1)
        out = out / 4.0
        out = out.transpose(1,2).reshape(B, C, H, W)
        return out + shortcut

class MambaEncoder(nn.Module):
    def __init__(self, in_ch=1, dims=(128,256,512,1024), depths=(2,2,15,2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, dims[0], 4, 4, bias=False), LayerNorm2d(dims[0]))
        self.stage1 = nn.Sequential(*[MambaBlock(dims[0]) for _ in range(depths[0])])
        self.down2  = nn.Sequential(LayerNorm2d(dims[0]), nn.Conv2d(dims[0], dims[1], 2, 2))
        self.stage2 = nn.Sequential(*[MambaBlock(dims[1]) for _ in range(depths[1])])
        self.down3  = nn.Sequential(LayerNorm2d(dims[1]), nn.Conv2d(dims[1], dims[2], 2, 2))
        self.stage3 = nn.Sequential(*[MambaBlock(dims[2]) for _ in range(depths[2])])
        self.down4  = nn.Sequential(LayerNorm2d(dims[2]), nn.Conv2d(dims[2], dims[3], 2, 2))
        self.stage4 = nn.Sequential(*[MambaBlock(dims[3]) for _ in range(depths[3])])

    def forward(self, x):
        x = self.stem(x)
        e1 = self.stage1(x)
        e2 = self.stage2(self.down2(e1))
        e3 = self.stage3(self.down3(e2))
        e4 = self.stage4(self.down4(e3))
        return e1, e2, e3, e4

class decoderBlock_conv(nn.Module):
    """Same conv decoder block as in MambaOut v2."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = conv2D_block(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class HybridMambaUNet(nn.Module):
    """Mamba UNet – conv decoder (v2)."""
    def __init__(self, in_ch=1, out_ch=4, dims=(128,256,512,1024), depths=(2,2,15,2)):
        super().__init__()
        self.encoder = MambaEncoder(in_ch, dims, depths)
        self.decoder4 = decoderBlock_conv(dims[3], dims[2], dims[2])
        self.decoder3 = decoderBlock_conv(dims[2], dims[1], dims[1])
        self.decoder2 = decoderBlock_conv(dims[1], dims[0], dims[0])
        self.up1 = nn.ConvTranspose2d(dims[0], 96, 2, 2)
        self.conv_up1 = conv2D_block(96, 64)
        self.up2 = nn.ConvTranspose2d(64, 24, 2, 2)
        self.head = nn.Conv2d(24, out_ch, 1)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d1 = self.decoder4(e4, e3)
        d2 = self.decoder3(d1, e2)
        d3 = self.decoder2(d2, e1)
        x = self.up1(d3)
        x = self.conv_up1(x)
        x = self.up2(x)
        return self.head(x)