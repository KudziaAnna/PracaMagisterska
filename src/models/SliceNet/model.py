"""
Basic UNet-2D implementation
"""
import torch.nn as nn
import models.SliceNet.model_comp as u_comp


class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.double_conv = u_comp.DoubleConv(n_channels, 64)

        self.down1 = u_comp.Down(64, 128)
        self.down2 = u_comp.Down(128, 256)
        self.down3 = u_comp.Down(256, 512)
        self.down4 = u_comp.Down(512, 1024)

        self.up4 = u_comp.Up(1024, 512)
        self.up3 = u_comp.Up(512, 256)
        self.up2 = u_comp.Up(256, 128)
        self.up1 = u_comp.Up(128, 64)
        self.conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.conv(x)

        return nn.functional.softmax(out, dim=1)


