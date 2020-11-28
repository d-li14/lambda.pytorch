import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, d, dk=16, du=1, Nh=4, m=None, r=23, stride=1):
        super(LambdaLayer, self).__init__()
        self.d = d
        self.dk = dk
        self.du = du
        self.Nh = Nh
        assert d % Nh == 0, 'd should be divided by Nh'
        dv = d // Nh
        self.dv = dv
        assert stride in [1, 2]
        self.stride = stride

        self.conv_qkv = nn.Conv2d(d, Nh * dk + dk * du + dv * du, 1, bias=False)
        self.norm_q = nn.BatchNorm2d(Nh * dk)
        self.norm_v = nn.BatchNorm2d(dv * du)
        self.softmax = nn.Softmax(dim=-1)
        self.lambda_conv = nn.Conv3d(du, dk, (1, r, r), padding = (0, (r - 1) // 2, (r - 1) // 2))

        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        N, C, H, W = x.shape

        qkv = self.conv_qkv(x)
        q, k, v = torch.split(qkv, [self.Nh * self.dk, self.dk * self.du, self.dv * self.du], dim=1)
        q = self.norm_q(q).view(N, self.Nh, self.dk, H*W)
        v = self.norm_v(v).view(N, self.du, self.dv, H*W)
        k = self.softmax(k.view(N, self.du, self.dk, H*W))

        lambda_c = torch.einsum('bukm,buvm->bkv', k, v)
        yc = torch.einsum('bhkm,bkv->bhvm', q, lambda_c)
        lambda_p = self.lambda_conv(v.view(N, self.du, self.dv, H, W)).view(N, self.dk, self.dv, H*W)
        yp = torch.einsum('bhkm,bkvm->bhvm', q, lambda_p)
        out = (yc + yp).reshape(N, C, H, W)

        if self.stride > 1:
            out = self.avgpool(out)

        return out
