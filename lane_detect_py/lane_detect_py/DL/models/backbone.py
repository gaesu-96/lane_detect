import torch
import torch.nn as nn

def channel_shuffle(x, groups=2):
    batch, channel, height, width = x.size()
    channel_per_groups = channel // groups

    x = x.view(batch, groups, channel_per_groups, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch, channel, height, width)

    return x

class ShuffleNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ShuffleNetBlock, self).__init__()

        self.stride = stride
        mid_channel = out_channel // 2
        self.group_norm = 2  # Number of groups for GroupNorm

        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, groups=in_channel, bias=False),
                nn.GroupNorm(self.group_norm, in_channel),
                nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(self.group_norm, mid_channel),
                nn.ReLU()
            )
        else:
            self.branch1 = None

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel if stride == 2 else mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(self.group_norm, mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1, groups=mid_channel, bias=False),
            nn.GroupNorm(self.group_norm, mid_channel),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(self.group_norm, mid_channel),
            nn.ReLU()
        )

    def forward(self, x):
        if self.stride == 2:
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            output = torch.cat([branch1, branch2], dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            branch2 = self.branch2(x2)
            output = torch.cat([x1, branch2], dim=1)

        output = channel_shuffle(output)
        return output

class shufflenet_v2(nn.Module):
    def __init__(self, stage_out_channels):
        super(shufflenet_v2, self).__init__()
        self.group_norm = 2  # Number of groups for GroupNorm

        self.fisrt = nn.Sequential(
            nn.Conv2d(3, stage_out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(self.group_norm, stage_out_channels[0]),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(stage_out_channels[0], stage_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(self.group_norm, stage_out_channels[0]),
            nn.ReLU(),
        )

        self.stage2 = self.make_stage(stage_out_channels[0], stage_out_channels[1], 2)
        self.stage3 = self.make_stage(stage_out_channels[1], stage_out_channels[2], 2)
        self.stage4 = self.make_stage(stage_out_channels[2], stage_out_channels[3], 2)

    def make_stage(self, in_channel, out_channel, repeat):
        layers = []
        layers.append(ShuffleNetBlock(in_channel, out_channel, stride=2))
        for i in range(repeat - 1):
            layers.append(ShuffleNetBlock(out_channel, out_channel, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        first = self.fisrt(x)
        conv1 = self.conv1(first)  # 1/2 180
        stage2 = self.stage2(conv1) # 1/4 90
        stage3 = self.stage3(stage2) # 1/8 45
        stage4 = self.stage4(stage3) # 1/16 22

        return first, conv1, stage2, stage3, stage4
