import torch
import torch.nn as nn
from .layers import LayerNorm2d, GatedCNNBlock
from .hybrid_mambaout_unet import MambaOutEncoder

# But v3 uses a different decoderBlock: uses GatedCNNBlock instead of conv2D_block
class decoderBlock_Gated(nn.Module):
    """Decoder block with GatedCNNBlock stages (depth=2)."""
    def __init__(self, in_ch, skip_ch, out_ch, depth=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False),
            LayerNorm2d(out_ch)
        )
        self.stage = nn.Sequential(*[GatedCNNBlock(out_ch) for _ in range(depth)])

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.stage(x)

class PureMambaOutUNet(nn.Module):
    """MambaOut UNet – GatedCNN decoder (v3). Input (B,1,224,224) → (B,4,224,224)."""
    def __init__(self, in_ch=1, out_ch=4, dims=(128,256,512,1024), depths=(2,2,15,2)):
        super().__init__()
        self.encoder = MambaOutEncoder(in_ch, dims, depths)
        self.decoder4 = decoderBlock_Gated(dims[3], dims[2], dims[2])
        self.decoder3 = decoderBlock_Gated(dims[2], dims[1], dims[1])
        self.decoder2 = decoderBlock_Gated(dims[1], dims[0], dims[0])
        # tail also uses GatedCNNBlock
        self.up1 = nn.ConvTranspose2d(dims[0], 96, 2, stride=2)
        self.norm_up1 = LayerNorm2d(96)
        self.gated_up1 = GatedCNNBlock(96)
        self.up2 = nn.ConvTranspose2d(96, 24, 2, stride=2)
        self.norm_up2 = LayerNorm2d(24)
        self.gated_up2 = GatedCNNBlock(24)
        self.head = nn.Conv2d(24, out_ch, 1)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d1 = self.decoder4(e4, e3)
        d2 = self.decoder3(d1, e2)
        d3 = self.decoder2(d2, e1)
        x = self.gated_up1(self.norm_up1(self.up1(d3)))
        x = self.gated_up2(self.norm_up2(self.up2(x)))
        return self.head(x)