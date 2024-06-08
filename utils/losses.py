import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from typing import Tuple

def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d

def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        loss = -(self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean())
        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, reduction: str = 'mean', max_val: float = 1.0) -> None:
        super(SSIMLoss, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.padding: int = self.compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    @staticmethod
    def compute_zero_padding(kernel_size: int) -> int:
        """Computes zero padding."""
        return (kernel_size - 1) // 2

    def filter2D(
            self,
            input: torch.Tensor,
            kernel: torch.Tensor,
            channel: int) -> torch.Tensor:
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # compute local mean per channel
        mu1: torch.Tensor = self.filter2D(img1, kernel, c)
        mu2: torch.Tensor = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss
# ------------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


##MSSSIM+L1 Loss
class MS_SSIM_L1_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=255.0,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=0.025,  # weight of ssim and l1 loss
                 compensation=200.0,  # final factor for total loss
                 cuda_dev=0,  # cuda device choice
                 channel=3):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel

        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
        # average l1 loss in num channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
