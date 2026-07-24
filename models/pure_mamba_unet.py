import torch
import torch.nn as nn
from .layers import LayerNorm2d
from .hybrid_mamba_unet import MambaBlock   # reuse the MambaBlock

class decoderBlock_Mamba(nn.Module):
    """Decoder block that uses MambaBlock stages (depth=2)."""
    def __init__(self, in_ch, skip_ch, out_ch, depth=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False),
            LayerNorm2d(out_ch)
        )
        self.stage = nn.Sequential(*[MambaBlock(out_ch) for _ in range(depth)])

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.stage(x)

class PureMambaUNet(nn.Module):
    """Symmetric Mamba UNet – encoder and decoder both use MambaBlock."""
    def __init__(self, in_ch=1, out_ch=4, dims=(128,256,512,1024), depths=(2,2,15,2)):
        super().__init__()
        # Reuse the MambaEncoder from v2 (or define again)
        from .hybrid_mamba_unet  import MambaEncoder
        self.encoder = MambaEncoder(in_ch, dims, depths)
        self.decoder4 = decoderBlock_Mamba(dims[3], dims[2], dims[2])
        self.decoder3 = decoderBlock_Mamba(dims[2], dims[1], dims[1])
        self.decoder2 = decoderBlock_Mamba(dims[1], dims[0], dims[0])
        # tail also uses MambaBlock
        self.up1 = nn.ConvTranspose2d(dims[0], 96, 2, stride=2)
        self.norm_up1 = LayerNorm2d(96)
        self.mamba_up1 = MambaBlock(96)
        self.up2 = nn.ConvTranspose2d(96, 24, 2, stride=2)
        self.norm_up2 = LayerNorm2d(24)
        self.mamba_up2 = MambaBlock(24)
        self.head = nn.Conv2d(24, out_ch, 1)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d1 = self.decoder4(e4, e3)
        d2 = self.decoder3(d1, e2)
        d3 = self.decoder2(d2, e1)
        x = self.mamba_up1(self.norm_up1(self.up1(d3)))
        x = self.mamba_up2(self.norm_up2(self.up2(x)))
        return self.head(x)