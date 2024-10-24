import torch
import torch.nn as nn
from .backbone import shufflenet_v2
from .decoder import Decoder

class SegmentShufflenetV2(nn.Module):
    def __init__(self, num_classes):
        super(SegmentShufflenetV2, self).__init__()

        self.stage_out_channels = [64, 128, 256, 512]
        
        self.shufflenet = shufflenet_v2(self.stage_out_channels)
        self.decoder = Decoder(self.stage_out_channels, self.shufflenet, num_classes)

    def forward(self, x):
        output = self.decoder(x)

        return output
    

class Unet(nn.Module):
  def __init__(self, num_classes):
    super(Unet, self).__init__()

    self.down_conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )

    self.pool1 = nn.Sequential(
        nn.MaxPool2d(2, stride = 2)
    )

    self.down_conv2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    self.pool2 = nn.Sequential(
        nn.MaxPool2d(2, stride = 2)
    )
    
    self.down_conv3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
    )

    self.pool3 = nn.Sequential(
        nn.MaxPool2d(2, stride = 2)
    )
    
    self.down_conv4 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
    )

    self.pool4 = nn.Sequential(
        nn.MaxPool2d(2, stride = 2)
    )
    
    self.bottle_conv = nn.Sequential(
        nn.Conv2d(512, 1024, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.Conv2d(1024, 512, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
    )

    self.up_conv4 = nn.Sequential(
        nn.ConvTranspose2d(512, 512, kernel_size = 2, stride = 2, padding = 0)
    )

    self.up_conv4_1 = nn.Sequential(
        nn.Conv2d(1024, 512, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 256, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
    )

    self.up_conv3 = nn.Sequential(
        nn.ConvTranspose2d(256, 256, kernel_size = 2, stride = 2, padding = 0)
    )

    self.up_conv3_1 = nn.Sequential(
        nn.Conv2d(512, 256, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    )

    self.up_conv2 = nn.Sequential(
        nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2, padding = 0)
    )

    self.up_conv2_1 = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )

    self.up_conv1 = nn.Sequential(
        nn.ConvTranspose2d(64, 64, kernel_size = 2, stride = 2, padding = 0)
    )

    self.up_conv1_1 = nn.Sequential(
        nn.Conv2d(128, 64, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    
    self.output = nn.Sequential(
        nn.Conv2d(64, num_classes, kernel_size = 1, padding = 0, stride = 1)
    )

  def forward(self, x):
    down1 = self.down_conv1(x) # 640
    pool1 = self.pool1(down1) #320
    down2 = self.down_conv2(pool1) #320
    pool2 = self.pool2(down2) #160
    down3 = self.down_conv3(pool2) #160
    pool3 = self.pool3(down3) #80
    down4 = self.down_conv4(pool3) #80
    pool4 = self.pool4(down4) #40
    bottle = self.bottle_conv(pool4) #40
    up4 = self.up_conv4(bottle) #80
    concat4 = torch.concat((down4, up4), dim=1)
    up4_1 = self.up_conv4_1(concat4)
    up3 = self.up_conv3(up4_1)
    concat3 = torch.concat((down3, up3), dim=1)
    up3_1 = self.up_conv3_1(concat3)
    up2 = self.up_conv2(up3_1)
    concat2 = torch.concat((down2, up2), dim=1)
    up2_1 = self.up_conv2_1(concat2)
    up1 = self.up_conv1(up2_1)
    concat1 = torch.concat((down1, up1), dim=1)
    up1_1 = self.up_conv1_1(concat1)
    output = self.output(up1_1)


    return output