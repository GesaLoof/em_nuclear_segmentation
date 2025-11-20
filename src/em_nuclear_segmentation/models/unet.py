import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_prob=0.0):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c, dropout=0.0, norm_layer=nn.BatchNorm2d):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                norm_layer(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                norm_layer(out_c),
                nn.ReLU(inplace=True)
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            return nn.Sequential(*layers)

        # Encoder (no dropout here by default)
        self.encoder = nn.ModuleList([
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        ])

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck (dropout optional)
        self.bottleneck = conv_block(512, 1024, dropout=dropout_prob)

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
        ])

        # Decoder (apply dropout here)
        self.decoder = nn.ModuleList([
            conv_block(1024, 512, dropout=dropout_prob),
            conv_block(512, 256, dropout=dropout_prob),
            conv_block(256, 128, dropout=dropout_prob),
            conv_block(128, 64, dropout=dropout_prob),
        ])

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[i](x)

        return self.output(x)
