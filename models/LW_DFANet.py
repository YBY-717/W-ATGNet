import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RefinementNet import RefinementNet
from models.StreamA_HybridUNet import StreamA_HybridUNet
from models.StreamB_PredictionThreshold import StreamB_HighFreq
from utils.Wavelet import HaarWavelet

# 可学习逆小波变换
class LearnableIDWT(nn.Module):
    def __init__(self, in_channels=12, out_channels=3):
        super(LearnableIDWT, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.reconstruct = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, ll, high):
        x = torch.cat([ll, high], dim=1)
        x = self.fusion(x)
        return self.reconstruct(x)

class LW_DFANet(nn.Module):
    def __init__(self):
        super(LW_DFANet, self).__init__()

        # 1. 小波变换工具 (这里用简单的 Haar 卷积实现前向)
        self.wavelet = HaarWavelet()

        # 2. Stream B: 高频净化流
        # 输入: High(9) + LL(3) -> 输出: Clean_High(9), Attn_Map(1)
        self.stream_b = StreamB_HighFreq(high_channels=9, low_channels=3)

        # 3. Stream A: 低频恢复流
        # 输入: LL(3), Attn_Map(1) -> 输出: Restored_LL(3)
        self.stream_a = StreamA_HybridUNet(in_channels=3)

        # 4. 逆变换重构 (IDWT 替代方案)
        # 输入: Restored_LL(3) + Clean_High(9) = 12 通道
        # 输出: Coarse_Image(3)
        self.idwt_learnable = LearnableIDWT(in_channels=12, out_channels=3)

        # 5. 全局细化 (Refinement)
        # 输入: Coarse(3) -> 输出: Final(3)
        self.refinement = RefinementNet(in_channels=3)

    def forward(self, x, return_loss_components=False):

        # === Stage 1: 分解 (Decomposition) ===
        # LL: [B, 3, H/2, W/2]
        # High: [B, 9, H/2, W/2]
        ll_raw, high_raw = self.wavelet.dwt(x)

        # === Stage 2: 高频流 (Stream B) ===
        # 利用 LL 的亮度信息，净化 High，并生成 Attn
        # f_clean: [B, 9, H/2, W/2] (用于重构)
        # attn_map: [B, 1, H/2, W/2] (用于引导)
        # gate_mask: 用于计算稀疏 Loss (可选)
        f_clean, attn_map, gate_mask = self.stream_b(high_raw, ll_raw)

        # === Stage 3: 低频流 (Stream A) ===
        # 恢复色彩和去雾，同时接受 Attn 保护边缘
        # ll_restored: [B, 3, H/2, W/2]
        ll_restored = self.stream_a(ll_raw, attn_map)

        # === Stage 4: 重构 (Reconstruction) ===
        # 逆变换 (Learnable IDWT)
        i_coarse = self.idwt_learnable(ll_restored, f_clean)  # [B, 3, H, W]

        # === Stage 5: 细化 (Refinement) ===
        # 消除伪影，最终修饰
        i_final = self.refinement(i_coarse)  # [B, 3, H, W]

        if return_loss_components:
            return i_final, f_clean, ll_restored, attn_map, gate_mask
        else:
            return i_final


if __name__ == "__main__":
    # 1. 实例化模型
    model = LW_DFANet()

    # 2. 创建模拟输入 (Batch=2, RGB, 256x256)
    dummy_input = torch.randn(2, 3, 256, 256)

    # 3. 运行前向传播 (训练模式，查看中间变量)
    final_img, f_clean, ll_restored, attn = model(dummy_input, return_loss_components=True)

    # 4. 打印尺寸验证
    print("=== LW-DFANet 模型构建成功 ===")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"1. 最终输出 (Final): {final_img.shape} -> 应为 [2, 3, 256, 256]")
    print(f"2. 净化高频 (High):  {f_clean.shape}  -> 应为 [2, 9, 128, 128]")
    print(f"3. 恢复低频 (Low):   {ll_restored.shape}-> 应为 [2, 3, 128, 128]")
    print(f"4. 注意力图 (Attn):  {attn.shape}     -> 应为 [2, 1, 128, 128]")