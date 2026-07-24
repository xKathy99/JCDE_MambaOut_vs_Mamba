import torch
import torch.nn as nn
import timm

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

class decoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = conv2D_block(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))

class pretrainedMambaOutEncUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, model_name='mambaout_base.in1k', pretrained=True):
        super().__init__()
        # Load encoder from timm, output NHWC feature maps
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=in_ch)
        enc_ch = self.encoder.feature_info.channels()   # e.g. [64, 128, 256, 512] for base
        # Decoder (same as in notebook)
        self.decoder4 = decoderBlock(enc_ch[3], enc_ch[2], enc_ch[2])
        self.decoder3 = decoderBlock(enc_ch[2], enc_ch[1], enc_ch[1])
        self.decoder2 = decoderBlock(enc_ch[1], enc_ch[0], enc_ch[0])
        self.up1 = nn.ConvTranspose2d(enc_ch[0], 96, 2, stride=2)
        self.conv1 = conv2D_block(96, 64)
        self.up2 = nn.ConvTranspose2d(64, 24, 2, stride=2)
        self.head = nn.Conv2d(24, out_ch, 1)

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x: (B, C, H, W) with C=1 (grayscale)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)   # repeat to 3 channels (MambaOut expects 3)
        features = self.encoder(x)       # list of NHWC tensors
        # Convert NHWC -> NCHW
        e1, e2, e3, e4 = [f.permute(0,3,1,2).contiguous() for f in features]
        d1 = self.decoder4(e4, e3)
        d2 = self.decoder3(d1, e2)
        d3 = self.decoder2(d2, e1)
        out = self.head(self.up2(self.conv1(self.up1(d3))))
        return out