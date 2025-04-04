from torchvision.models import vgg13, VGG13_Weights
import torch
import torch.nn.functional as Fun
import torch.nn as nn

class VGG13Encoder(torch.nn.Module):
    def __init__(self, num_blocks, weights=VGG13_Weights.DEFAULT):
        super().__init__()
        self.num_blocks = num_blocks
        feature_extractor = vgg13(weights=weights).features
        self.blocks = torch.nn.ModuleList()
        for idx in range(self.num_blocks):
            self.blocks.append(nn.Sequential(
                feature_extractor[idx*5:idx*5 + 4])
            )
    def forward(self, X):
        activations = []
        for idx, block in enumerate(self.blocks):
            X = block(X)
            activations.append(X)
            X = torch.functional.F.max_pool2d(X, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        return activations

class DecoderBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.upconv = torch.nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.relu = torch.nn.ReLU()
    def forward(self, down, left):
        x = Fun.interpolate(down, scale_factor=2, mode='nearest')
        x = self.upconv(x)
        x = torch.cat([x, left], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class Decoder(torch.nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, DecoderBlock(num_filters * 2 ** idx))
    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up

class UNet(torch.nn.Module):
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()
        self.encoder = VGG13Encoder(num_blocks)
        self.decoder = Decoder(64, num_blocks-1)
        self.final = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1
        )
    def forward(self, x):
        acts = self.encoder.forward(x)
        x = self.decoder.forward(acts)
        x = self.final(x)
        return x