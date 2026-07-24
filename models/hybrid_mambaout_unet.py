import torch
import torch.nn as nn
from .layers import LayerNorm2d, GatedCNNBlock, conv2D_block

class MambaOutEncoder(nn.Module):
    """MambaOut encoder with depths (2,2,15,2) – same as notebook."""
    def __init__(self, in_ch=1, dims=(128,256,512,1024), depths=(2,2,15,2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 4, stride=4, bias=False),
            LayerNorm2d(dims[0])
        )
        self.stage1 = nn.Sequential(*[GatedCNNBlock(dims[0]) for _ in range(depths[0])])
        self.down2  = nn.Sequential(LayerNorm2d(dims[0]), nn.Conv2d(dims[0], dims[1], 2, 2))
        self.stage2 = nn.Sequential(*[GatedCNNBlock(dims[1]) for _ in range(depths[1])])
        self.down3  = nn.Sequential(LayerNorm2d(dims[1]), nn.Conv2d(dims[1], dims[2], 2, 2))
        self.stage3 = nn.Sequential(*[GatedCNNBlock(dims[2]) for _ in range(depths[2])])
        self.down4  = nn.Sequential(LayerNorm2d(dims[2]), nn.Conv2d(dims[2], dims[3], 2, 2))
        self.stage4 = nn.Sequential(*[GatedCNNBlock(dims[3]) for _ in range(depths[3])])

    def forward(self, x):
        x = self.stem(x)
        e1 = self.stage1(x)
        e2 = self.stage2(self.down2(e1))
        e3 = self.stage3(self.down3(e2))
        e4 = self.stage4(self.down4(e3))
        return e1, e2, e3, e4

class decoderBlock(nn.Module):
    """Decoder block with conv2D_block (v2 style)."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = conv2D_block(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class HybridMambaOutUNet(nn.Module):
    """MambaOut UNet – conv decoder (v2). Input (B,1,224,224) → (B,4,224,224)."""
    def __init__(self, in_ch=1, out_ch=4, dims=(128,256,512,1024), depths=(2,2,15,2)):
        super().__init__()
        self.encoder = MambaOutEncoder(in_ch, dims, depths)
        self.decoder4 = decoderBlock(dims[3], dims[2], dims[2])
        self.decoder3 = decoderBlock(dims[2], dims[1], dims[1])
        self.decoder2 = decoderBlock(dims[1], dims[0], dims[0])
        self.up1 = nn.ConvTranspose2d(dims[0], 96, 2, stride=2)
        self.conv_up1 = conv2D_block(96, 64)
        self.up2 = nn.ConvTranspose2d(64, 24, 2, stride=2)
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