import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()  # PReLU 在图像复原中通常优于 ReLU
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ColorAttention(nn.Module):

    def __init__(self, channels, reduction=4):
        super(ColorAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class TransformerBottleneck(nn.Module):

    def __init__(self, channels, num_heads=4, ff_dim=128):
        super(TransformerBottleneck, self).__init__()
        self.channels = channels
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, channels)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.flatten(2)  # [B, C, H*W]
        x_flat = x_flat.transpose(1, 2)  # [B, H*W, C]

        x_res = x_flat
        x_flat = self.norm1(x_flat)
        attn_out, _ = self.attn(
            x_flat,
            x_flat,
            x_flat
        )
        x_flat = x_res + attn_out


        x_res = x_flat
        x_flat = self.norm2(x_flat)
        ffn_out = self.ffn(x_flat)
        x_flat = x_res + ffn_out


        x_out = x_flat.transpose(1, 2)
        x_out = x_out.reshape(b, c, h, w)
        return x_out


class AttentionModulation(nn.Module):

    def __init__(self):
        super(AttentionModulation, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_low, attn_map):
        return f_low * (1 + self.alpha * attn_map)


# ==========================================
# 主网络: Stream A (低频恢复流)
# ==========================================

class StreamA_HybridUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(StreamA_HybridUNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels)
        )
        self.pool1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        self.enc2 = ResBlock(base_channels * 2)
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)


        self.bottleneck_trans = TransformerBottleneck(base_channels * 4)
        self.bottleneck_color = ColorAttention(base_channels * 4)


        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce2 = nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, 1)
        self.dec2 = ResBlock(base_channels * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce1 = nn.Conv2d(base_channels * 2 + base_channels, base_channels, 1)

        self.modulate = AttentionModulation()
        self.dec1 = ResBlock(base_channels)

        self.final_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, attn_map_from_stream_b):
        # === Encoding ===
        e1 = self.enc1(x)  # [B, 32, 128, 128]
        p1 = self.pool1(e1)  # [B, 64, 64, 64]

        e2 = self.enc2(p1)  # [B, 64, 64, 64]
        p2 = self.pool2(e2)  # [B, 128, 32, 32]

        # === Bottleneck Processing ===
        b = self.bottleneck_trans(p2)
        b = self.bottleneck_color(b)

        # === Decoding ===
        # Layer 2
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)  # Skip Connection
        d2 = self.reduce2(d2)
        d2 = self.dec2(d2)

        # Layer 1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip Connection
        d1 = self.reduce1(d1)

        # === Cross-Stream Interaction ===
        d1_mod = self.modulate(d1, attn_map_from_stream_b)

        # === Final Polish ===
        out_feat = self.dec1(d1_mod)
        out = self.final_conv(out_feat)

        return out


# 测试代码
if __name__ == "__main__":
    model = StreamA_HybridUNet()
    dummy_input = torch.randn(1, 3, 128, 128)
    dummy_attn = torch.randn(1, 1, 128, 128)

    output = model(dummy_input, dummy_attn)
    print(f"Stream A Output Shape: {output.shape}")
    print("Stream A Build Success!")