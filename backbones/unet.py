import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Encoder', 'Decoder']


class EmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(EmbeddingBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(DownBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)

        self.block_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.Dropout2d(drop),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.Dropout2d(drop)
        )
        self.block_mp = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_in):
        x1 = self.block_conv(x_in)
        x1_pool = self.block_mp(x1)
        return x1, x1_pool


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0, act=None):
        super(UpBlock, self).__init__()
        self.act = act
        padding = int((kernel_size - 1) / 2)

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.Dropout2d(drop),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.Dropout2d(drop),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x_in, x_up):
        x_cat = torch.cat((x_in, x_up), 1)
        x = self.block(x_cat)

        if self.act is not None:
            x = getattr(F, self.act)(x)

        return x


class Encoder(nn.Module):
    def __init__(self, enc_dim, drop=0):
        super(Encoder, self).__init__()
        self.dims = [enc_dim // 4, enc_dim // 2, enc_dim, enc_dim]
        self.down1 = DownBlock(3, self.dims[-4], drop=drop)
        self.down2 = DownBlock(self.dims[-4], self.dims[-3], drop=drop)
        self.down3 = DownBlock(self.dims[-3], self.dims[-2], drop=drop)

    def forward(self, x_in):
        (x1, x1_pool) = self.down1(x_in)
        (x2, x2_pool) = self.down2(x1_pool)
        (x3, x3_pool) = self.down3(x2_pool)
        return x1, x2, x3, x3_pool


class Decoder(nn.Module):
    def __init__(self, res_dims, drop=0):
        super(Decoder, self).__init__()
        self.embedding = EmbeddingBlock(res_dims[-1], res_dims[-1], drop=drop)
        self.up3 = UpBlock(2 * res_dims[-1], res_dims[-2], act='relu', drop=drop)
        self.up2 = UpBlock(2 * res_dims[-2], res_dims[-3], act='relu', drop=drop)
        self.up1 = UpBlock(2 * res_dims[-3], 2, act=None, drop=drop)

    def forward(self, features):
        assert type(features) in (tuple, list), 'x must be a list'

        x_emb = self.embedding(features[-1])
        x = self.up3(x_emb, features[-1])
        x = self.up2(x, features[-2])
        x = self.up1(x, features[-3])

        return x