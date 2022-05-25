import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channnels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channnels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channnels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channnels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channnels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channnels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channnels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channnels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channnels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channnels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channnels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channnels=1024, out_channels=512)

        self.uppool4 = nn.ConvTranspose2d(
            in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channnels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channnels=512, out_channels=256)

        self.uppool3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channnels=2*256, out_channels=256)
        self.dec3_1 = CBR2d(in_channnels=256, out_channels=128)

        self.uppool2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channnels=2*128, out_channels=128)
        self.dec2_1 = CBR2d(in_channnels=128, out_channels=64)

        self.uppool1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channnels=2*64, out_channels=64)
        self.dec1_1 = CBR2d(in_channnels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        uppool4 = self.uppool4(dec5_1)

        dec4_2 = self.dec4_2(torch.cat((uppool4, enc4_2), dim=1))
        dec4_1 = self.dec4_1(dec4_2)

        uppool3 = self.uppool3(dec4_1)

        dec3_2 = self.dec3_2(torch.cat((uppool3, enc3_2), dim=1))
        dec3_1 = self.dec3_1(dec3_2)

        uppool2 = self.uppool2(dec3_1)

        dec2_2 = self.dec2_2(torch.cat((uppool2, enc2_2), dim=1))
        dec2_1 = self.dec2_1(dec2_2)

        uppool1 = self.uppool1(dec2_1)

        dec1_2 = self.dec1_2(torch.cat((uppool1, enc1_2), dim=1))
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x
