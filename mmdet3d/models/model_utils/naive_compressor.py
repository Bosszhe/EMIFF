import torch
import torch.nn as nn


class NaiveCompressor(nn.Module):
    def __init__(self, input_dim, compress_ratio):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//compress_ratio, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim//compress_ratio, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim//compress_ratio, input_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3,
                           momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class upsample(nn.Module):

    def __init__(self, if_deconv=True, channels=None):
        super(upsample, self).__init__()
        if if_deconv:
            self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)

        return x


class downsample(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(downsample, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class NaiveCompressor_UNet(nn.Module):
    def __init__(self, input_dim, c_compress_ratio, s_compress_ratio):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//c_compress_ratio, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim//c_compress_ratio, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.final_channels = input_dim//c_compress_ratio
        self.downsamples = nn.Sequential()
        for i in range(s_compress_ratio):
            module_name = f'down{i}'
            self.downsamples.add_module(module_name,downsample(self.final_channels,self.final_channels))

        self.upsamples = nn.Sequential()
        for i in range(s_compress_ratio):
            module_name = f'up{i}'
            self.upsamples.add_module(module_name, upsample(if_deconv=True, channels=self.final_channels))


        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim//c_compress_ratio, input_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3,
                           momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.downsamples(x)

        # from IPython import embed 
        # embed(header='s_compression')

        x = self.upsamples(x)
        x = self.decoder(x)

        return x