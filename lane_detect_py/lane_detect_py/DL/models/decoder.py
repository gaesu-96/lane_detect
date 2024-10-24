import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import channel_shuffle

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsampling, reduce=False):
        super(DecoderBlock, self).__init__()

        self.upsampling = upsampling
        self.reduce = reduce

        mid_channel = out_channel // 2
        self.group_norm = 2  # Number of groups for GroupNorm

        if upsampling:
            self.branch1 = nn.Sequential(
                nn.ConvTranspose2d(in_channel, mid_channel, kernel_size=2, stride=2, padding=0, groups=2, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.ReLU()
            )
        else:
            if reduce:
                self.branch1 = nn.Sequential(
                    nn.Conv2d(in_channel // 2 if reduce else mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.GroupNorm(self.group_norm, mid_channel),
                    nn.ReLU(),
                    nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, groups=2, bias=False),
                    nn.GroupNorm(self.group_norm, mid_channel),
                    nn.ReLU()
                )
            else:
                self.branch1 = None

        if upsampling:
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.ReLU(),
                nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=2, stride=2, padding=0, groups=2, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.ReLU()
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channel // 2 if reduce else mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.ReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, groups=2, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.ReLU()
            )

    def forward(self, x):
        if self.upsampling:
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            output = torch.cat([branch1, branch2], dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            if self.reduce:
                x1 = self.branch1(x1)
            branch2 = self.branch2(x2)
            output = torch.cat([x1, branch2], dim=1)

        output = channel_shuffle(output)
        return output

class ChannelAttention(nn.Module):
    def __init__(self, input_channel):
        super(ChannelAttention, self).__init__()

        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.group_norm = 2  # Number of groups for GroupNorm

        self.block = nn.Sequential(
            nn.Conv2d(input_channel, input_channel // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(self.group_norm, input_channel // 4),
            nn.Conv2d(input_channel // 4, input_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(self.group_norm, input_channel),
            nn.ReLU()
        )

    def forward(self, x):
        avg = self.avepool(x)
        avg = self.block(avg)

        max = self.maxpool(x)
        max = self.block(max)

        concat = avg + max
        return F.sigmoid(concat)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.group_norm = 2  # Number of groups for GroupNorm
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        x = self.conv(x)
        return F.sigmoid(x)

class Decoder(nn.Module):
    def __init__(self, stage_out_channels, backbone, num_classes):
        super(Decoder, self).__init__()

        self.backbone = backbone

        self.up_stage3 = self.make_stage(stage_out_channels[3], stage_out_channels[2], 2)
        self.up_stage2 = self.make_stage(stage_out_channels[2], stage_out_channels[1], 2)
        self.up_stage1 = self.make_stage(stage_out_channels[1], stage_out_channels[0], 2)
        self.first_up_stage = self.make_stage(stage_out_channels[0], stage_out_channels[0], 2)

        self.reduce_channel3 = DecoderBlock(stage_out_channels[3], stage_out_channels[2], upsampling=False, reduce=True)
        self.reduce_channel2 = DecoderBlock(stage_out_channels[2], stage_out_channels[1], upsampling=False, reduce=True)
        self.reduce_channel1 = DecoderBlock(stage_out_channels[1], stage_out_channels[0], upsampling=False, reduce=True)
        self.first_reduce_channel = DecoderBlock(stage_out_channels[1], stage_out_channels[0], upsampling=False, reduce=True)

        self.channel_attention = ChannelAttention(stage_out_channels[0])
        self.spatial_attention = SpatialAttention()

        self.output = nn.Conv2d(stage_out_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def make_stage(self, in_channel, out_channel, repeat):
        layers = []
        layers.append(DecoderBlock(in_channel, out_channel, True))
        for i in range(repeat - 1):
            layers.append(DecoderBlock(out_channel, out_channel, False))
        return nn.Sequential(*layers)

    def forward(self, x):
        first, conv1, stage2, stage3, stage4 = self.backbone(x)
        up_stage3 = self.up_stage3(stage4)
        concat3 = torch.cat([up_stage3, stage3], dim=1)
        concat3 = channel_shuffle(concat3)
        reduce3 = self.reduce_channel3(concat3)

        up_stage2 = self.up_stage2(reduce3)
        concat2 = torch.cat([up_stage2, stage2], dim=1)
        concat2 = channel_shuffle(concat2)
        reduce2 = self.reduce_channel2(concat2)

        up_stage1 = self.up_stage1(reduce2)
        concat1 = torch.cat([up_stage1, conv1], dim=1)
        concat1 = channel_shuffle(concat1)
        reduce1 = self.reduce_channel1(concat1)

        first_up_stage = self.first_up_stage(reduce1)
        first_concat = torch.cat([first_up_stage, first], dim=1)
        first_concat = channel_shuffle(first_concat)
        first_reduce = self.first_reduce_channel(first_concat)

        channel_attention = self.channel_attention(first_reduce) * first_reduce
        spatial_attention = self.spatial_attention(channel_attention) * channel_attention
        output = first_reduce + spatial_attention

        output = self.output(output)
        return output
