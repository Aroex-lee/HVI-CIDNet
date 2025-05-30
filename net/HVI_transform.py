import torch
import torch.nn as nn
import math

pi = math.pi

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = nn.Parameter(torch.full([1], 0.2))  # 可学习的 k 参数
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0

    def HVIT(self, img):
        """
        RGB -> HVIT（色调-强度空间）
        输入: img [B, 3, H, W]
        输出: xyz [B, 3, H, W]
        """
        eps = 1e-8
        device = img.device
        dtype = img.dtype

        # 获取 H, W
        B, C, H, W = img.shape

        # 最大通道值、最小通道值 [B, H, W]
        value = img.max(dim=1)[0]
        img_min = img.min(dim=1)[0]

        # 初始化 hue
        hue = torch.zeros((B, H, W), device=device, dtype=dtype)

        # 避免除以 0
        diff = value - img_min + eps

        r, g, b = img[:, 0], img[:, 1], img[:, 2]

        # 按最大值通道分别计算 hue
        mask_r = (value == r)
        mask_g = (value == g)
        mask_b = (value == b)

        hue[mask_r] = ((g - b) / diff)[mask_r] % 6
        hue[mask_g] = 2.0 + ((b - r) / diff)[mask_g]
        hue[mask_b] = 4.0 + ((r - g) / diff)[mask_b]

        # 灰度值处理
        gray_mask = (value == img_min)
        hue[gray_mask] = 0.0

        hue = hue / 6.0  # 归一化到 [0, 1]

        # saturation & value
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0  # 饱和度处理

        # 升维以便后续拼接
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        # 颜色敏感度计算
        k = self.density_k
        self.this_k = k.item()
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)

        # HVI 计算
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value

        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def PHVIT(self, img):
        """
        HVIT -> RGB
        输入: img [B, 3, H, W]
        输出: rgb [B, 3, H, W]
        """
        eps = 1e-8
        H, V, I = img[:, 0], img[:, 1], img[:, 2]

        # clip 范围控制
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)
        v = I

        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)

        # 反算 hue 和 saturation
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1.0
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        # 反算 RGB（HSV -> RGB）
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - f * s)
        t = v * (1. - (1. - f) * s)

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0], g[hi0], b[hi0] = v[hi0], t[hi0], p[hi0]
        r[hi1], g[hi1], b[hi1] = q[hi1], v[hi1], p[hi1]
        r[hi2], g[hi2], b[hi2] = p[hi2], v[hi2], t[hi2]
        r[hi3], g[hi3], b[hi3] = p[hi3], q[hi3], v[hi3]
        r[hi4], g[hi4], b[hi4] = t[hi4], p[hi4], v[hi4]
        r[hi5], g[hi5], b[hi5] = v[hi5], p[hi5], q[hi5]

        rgb = torch.stack([r, g, b], dim=1)

        if self.gated2:
            rgb = rgb * self.alpha

        return rgb
