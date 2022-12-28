import torch
from torch import nn


class StyleMergeLayer(nn.Module):
    def __init__(self, channel, outchannel=512):
        super(StyleMergeLayer, self).__init__()
        self.cfcs = nn.ParameterList([nn.Parameter(torch.Tensor(channel, 2))])
        self.cfc = self.cfcs[0]
        self.cfc.data.fill_(0)

        self.relu = nn.PReLU()
        self.linear = nn.Linear(channel, outchannel)
        self.bn = nn.BatchNorm1d(outchannel)

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def forward(self, style):
        z = style * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)
        z = self.relu(z)
        z = self.linear(z)
        z = self.bn(z.unsqueeze(-1)).squeeze(-1)
        return z


class L2Norm(nn.Module):

    def __init__(self, dim=-1, eps=1e-5):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=self.dim, keepdim=True) + self.eps)


