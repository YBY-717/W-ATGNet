import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.Wavelet import HaarWavelet

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x, y):
        cos_sim = F.cosine_similarity(x, y, dim=1, eps=1e-8)
        loss = 1.0 - cos_sim.mean()
        return loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        # 计算水平和垂直方向的差分
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# Charbonnier Loss (平滑 L1)
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps**2))
        return loss

# Edge Loss (梯度损失)
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kw//2, kh//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        return filtered - new_filter

    def forward(self, x, y):
        return self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))


# Perceptual Loss (感知损失)
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False

        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),
            nn.Sequential(*list(vgg.children())[4:9]),
            nn.Sequential(*list(vgg.children())[9:18])
        ])
        self.l1 = nn.L1Loss()


    def forward(self, x, y):
        loss = 0
        x_feat, y_feat = x, y
        for block in self.blocks:
            x_feat = block(x_feat)
            y_feat = block(y_feat)
            loss += self.l1(x_feat, y_feat)
        return loss


class LW_DFANet_Loss(nn.Module):
    def __init__(self, wavelet_module=None, ablation_mode='full'):
        super(LW_DFANet_Loss, self).__init__()

        self.wavelet = HaarWavelet()  # 小波分解

        # 子损失函数
        self.char_loss = CharbonnierLoss()
        self.perc_loss = PerceptualLoss()
        self.edge_loss = EdgeLoss()
        self.tv_loss = TVLoss()
        self.color_loss = ColorLoss()

        # === 权重配置 (默认 Full 配置) ===
        self.lambda_rec = 1.0
        self.lambda_per = 1.0
        self.lambda_edge = 0.1
        self.lambda_color = 0.5
        self.lambda_tv = 0.1
        self.lambda_freq = 0.5
        self.lambda_gate = 0.01

        self.mode = ablation_mode
        print(f"Build Loss Function with mode: [{self.mode}]")

        if self.mode == 'wo_edge':
            self.lambda_edge = 0.0
            print(">>> Warning: Edge Loss is disabled!")
        
        elif self.mode == 'wo_tv':
            self.lambda_tv = 0.0
            print(">>> Warning: Tv Loss is disabled!")

    def forward(self, outputs, gt_img):
        i_final, f_clean, ll_restored, attn_map, gate_mask = outputs
        device = i_final.device

        # 1. 小波分解 GT
        with torch.no_grad():
            gt_ll, gt_high = self.wavelet.dwt(gt_img)

        # A. 全局重构损失 (必选)
        loss_rec = self.char_loss(i_final, gt_img)

        # B. 感知损失 (消融项)
        if self.lambda_per > 0:
            loss_per = self.perc_loss(i_final, gt_img)
        else:
            loss_per = torch.tensor(0.0, device=device)

        # C. 边缘损失(消融项)
        loss_edge = self.edge_loss(i_final, gt_img)

        # D. 频域一致性损失(必选)
        loss_low = self.char_loss(ll_restored, gt_ll)
        loss_high = self.char_loss(f_clean, gt_high)
        loss_freq = loss_low + loss_high

        # E. 门控稀疏损失(必选)
        loss_gate = torch.mean(gate_mask)

        # F. TV Loss(消融项)
        loss_tv = self.tv_loss(i_final)

        # G. Color Loss (消融项)
        if self.lambda_color > 0:
            loss_color = self.color_loss(i_final, gt_img)
        else:
            loss_color = torch.tensor(0.0, device=device)


        total_loss = (self.lambda_rec * loss_rec +
                      self.lambda_per * loss_per +
                      self.lambda_edge * loss_edge +
                      self.lambda_freq * loss_freq +
                      self.lambda_gate * loss_gate +
                      self.lambda_tv * loss_tv +
                      self.lambda_color * loss_color)

        return total_loss, {
            "rec": loss_rec.item(),
            "per": loss_per.item(),
            "edge": loss_edge.item(),
            "freq": loss_freq.item(),
            "gate": loss_gate.item(),
            "tv": loss_tv.item(),
            "color": loss_color.item()
        }