import torch
import torch.nn as nn
from transformers import AutoModel

class conv2D_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv2D = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        return self.conv2D(x)

class decoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = conv2D_block(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class pretrainedMambaEncUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, encoder_name="nvidia/MambaVision-T-1K"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
        # Freeze encoder initially
        for p in self.encoder.parameters():
            p.requires_grad = False

        # MambaVision-T channels
        enc_ch = [80, 160, 320, 640]

        self.decoder4 = decoderBlock(enc_ch[3], enc_ch[2], enc_ch[2])
        self.decoder3 = decoderBlock(enc_ch[2], enc_ch[1], enc_ch[1])
        self.decoder2 = decoderBlock(enc_ch[1], enc_ch[0], enc_ch[0])

        # Extra upsampling + head
        self.up5 = nn.ConvTranspose2d(enc_ch[0], 128, 2, stride=2)
        self.conv5 = conv2D_block(128, 64)
        self.up6 = nn.ConvTranspose2d(64, 24, 2, stride=2)
        self.head = nn.Conv2d(24, out_ch, kernel_size=1)

        # Alternative path (used in forward)
        self.UpSample2D_1 = nn.ConvTranspose2d(enc_ch[0], 96, 2, stride=2)
        self.Conv2D_1 = conv2D_block(96, 64)
        self.UpSample2D_2 = nn.ConvTranspose2d(64, 24, 2, stride=2)
        self.Conv2D_final = nn.Conv2d(24, out_ch, kernel_size=1)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)   # convert to 3-channel for pretrained encoder
        _, features = self.encoder(x)   # returns (?, features_list)
        e1, e2, e3, e4 = features       # shallow to deep

        d1 = self.decoder4(e4, e3)
        d2 = self.decoder3(d1, e2)
        d3 = self.decoder2(d2, e1)

        d5 = self.UpSample2D_1(d3)
        d5_1_1 = self.Conv2D_1(d5)
        d5_1 = self.UpSample2D_2(d5_1_1)
        out = self.Conv2D_final(d5_1)
        return out