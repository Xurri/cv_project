import torch
from torch import nn
import torch.nn.functional as F


# The argument n_class specifies the number of classes for the segmentation task.
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # -------
        # Каждый блок в encoder состоит из двух сверточных слоев,
        # за которыми следует max-pooling, за исключением последнего блока.
        # ------- 
        # Используем padding=1 как best practice и для сохранения размеров карт 
        # признаков после сверток, облегчения реализации skip-connection, 
        # убираем необходимость пост-обработки выходного изображения (размеров)
        # -------
        # input: 640x640x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 640x640x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 640x640x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 320x320x64

        # input: 320x320x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 320x320x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 320x320x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 160x160x128

        # input: 160x160x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 160x160x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 160x160x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 80x80x256

        # input: 80x80x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 80x80x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 80x80x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 40x40x512

        # input: 40x40x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 40x40x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 40x40x1024


        # Decoder
        # -------
        # Повышает размерность обратно до исходного изображения
        # Каждый блок в декодировщике состоит из слоя апсемплинга,
        # конкатенации с соответствующей картой признаков из encoder,
        # и двух сверточных слоев
        # -------
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # 40x40 -> 80x80
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1) # Concatenated: 512 + 512 = 1024
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 80x80 -> 160x160
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # Concatenated: 256 + 256 = 512
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 160x160 -> 320x320
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # Concatenated: 128 + 128 = 256
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 320x320 -> 640x640
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # Concatenated: 64 + 64 = 128
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1) # 640x640xn_class

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x)) # 640x640x64
        xe12 = F.relu(self.e12(xe11)) # 640x640x64
        xp1 = self.pool1(xe12) # 320x320x64

        xe21 = F.relu(self.e21(xp1)) # 320x320x128
        xe22 = F.relu(self.e22(xe21)) # 320x320x128
        xp2 = self.pool2(xe22)  # 160x160x128

        xe31 = F.relu(self.e31(xp2)) # 160x160x256
        xe32 = F.relu(self.e32(xe31)) # 160x160x256
        xp3 = self.pool3(xe32) # 80x80x256

        xe41 = F.relu(self.e41(xp3)) # 80x80x512
        xe42 = F.relu(self.e42(xe41)) # 80x80x512
        xp4 = self.pool4(xe42) # 40x40x512

        xe51 = F.relu(self.e51(xp4)) # 40x40x1024
        xe52 = F.relu(self.e52(xe51)) # 40x40x1024

        # Decoder
        xu1 = self.upconv1(xe52) # 80x80x512
        xu11 = torch.cat([xu1, xe42], dim=1) # 80x80x1024
        xd11 = F.relu(self.d11(xu11)) # 80x80x512
        xd12 = F.relu(self.d12(xd11)) # 80x80x512

        xu2 = self.upconv2(xd12) # 160x160x256
        xu22 = torch.cat([xu2, xe32], dim=1) # 160x160x512
        xd21 = F.relu(self.d21(xu22)) # 160x160x256
        xd22 = F.relu(self.d22(xd21)) # 160x160x256

        xu3 = self.upconv3(xd22) # 320x320x128
        xu33 = torch.cat([xu3, xe22], dim=1) # 320x320x256
        xd31 = F.relu(self.d31(xu33)) # 320x320x128
        xd32 = F.relu(self.d32(xd31)) # 320x320x128

        xu4 = self.upconv4(xd32) # 640x640x64
        xu44 = torch.cat([xu4, xe12], dim=1) # 640x640x128
        xd41 = F.relu(self.d41(xu44)) # 640x640x64
        xd42 = F.relu(self.d42(xd41)) # 640x640x64

        # Output layer
        out = self.outconv(xd42) # 640x640xn_class

        return out
