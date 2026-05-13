import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ThresholdPredictionNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=9, hidden_channels=32):
        super(ThresholdPredictionNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.body = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )

        self.tail = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        feat = self.head(x)
        feat = feat + self.body(feat)
        out = self.tail(feat)
        return self.softplus(out) + 1e-6

class SpatialAttentionGen(nn.Module):
    def __init__(self, in_channels=9):
        super(SpatialAttentionGen, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class StreamB_HighFreq(nn.Module):
    def __init__(self, high_channels=9, low_channels=3):
        super(StreamB_HighFreq, self).__init__()

        self.total_channels = high_channels + low_channels

        self.se_block = SEBlock(in_channels=self.total_channels)
        self.thresh_net = ThresholdPredictionNet(in_channels=self.total_channels, out_channels=high_channels)
        self.attn_gen = SpatialAttentionGen(in_channels=high_channels)

        self.k = 10.0

    def forward(self, i_high, i_low):

        f_joint = torch.cat([i_high, i_low], dim=1)  # [B, 12, H, W]
        f_weighted = self.se_block(f_joint)  # [B, 12, H, W]

        tau = self.thresh_net(f_weighted)  # [B, 9, H, W]

        # 核心公式: Gate = Sigmoid( k * (|x| - tau) )
        magnitude = torch.abs(i_high)
        gate = torch.sigmoid(self.k * (magnitude - tau))

        f_clean = i_high * gate  # [B, 9, H, W]

        attn_map = self.attn_gen(f_clean)  # [B, 1, H, W]

        return f_clean, attn_map, gate


# 测试代码
if __name__ == "__main__":
    dummy_high = torch.randn(2, 9, 128, 128)
    dummy_low = torch.randn(2, 3, 128, 128)

    model = StreamB_HighFreq()
    f_clean, attn_map = model(dummy_high, dummy_low)

    print(f"净化高频尺寸: {f_clean.shape}")
    print(f"注意力图尺寸: {attn_map.shape}")