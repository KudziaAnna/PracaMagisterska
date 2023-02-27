"""
Basic UNet-3D implementation
"""
import torch
import torch.nn as nn
import models.UNet3D.model_comp as u_comp
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, n_features):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features

        self.double_conv = u_comp.DoubleConv(n_channels, n_features)

        self.down1 = u_comp.Down(n_features, 2 * n_features)
        self.down2 = u_comp.Down(2 * n_features, 4 * n_features)
        self.down3 = u_comp.Down(4 * n_features, 8 * n_features)
        self.down4 = u_comp.Down(8 * n_features, 16 * n_features)

        self.up4 = u_comp.Up(16 * n_features, 8 * n_features)
        self.up3 = u_comp.Up(8 * n_features, 4 * n_features)
        self.up2 = u_comp.Up(4 * n_features, 2 * n_features)
        self.up1 = u_comp.Up(2 * n_features, n_features)
        self.conv = nn.Conv3d(n_features, n_classes, kernel_size=1)

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

        return out


if __name__ == "__main__":
    image = torch.rand((1, 1, 128, 128, 128))
    model = UNet3D(1, 8, 32)

    # print(model)
    #print(model.forward(x=image).shape)
    # pred = F.softmax(image, dim=1)
    # print(pred)
    # print(pred.shape)
    # pred = torch.argmax(pred, dim=1).squeeze(1)
    # print(pred)
    # print(pred.shape)