import torch
import torch.nn as nn

class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.act(out)

class RefinementNet(nn.Module):

    def __init__(self, in_channels=3, mid_channels=64, num_blocks=4):
        super(RefinementNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.body = nn.Sequential(
            *[ResBlock(mid_channels) for _ in range(num_blocks)]
        )

        self.tail = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):

        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)
        out = x + residual

        return out


if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 256, 256)
    model = RefinementNet()
    output = model(dummy_input)
    print(f"RefinementNet Output Shape: {output.shape}")