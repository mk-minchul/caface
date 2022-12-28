import torch

def cossim_loss(x, y, detach_y=True, average=True):
    assert x.ndim == 2
    assert y.ndim == 2
    if detach_y:
        y = y.detach()
    x = x / (torch.norm(x, 2, dim=-1, keepdim=True)+1e-5)
    y = y / (torch.norm(y, 2, dim=-1, keepdim=True)+1e-5)
    cossim = (x * y).sum(dim=-1)
    loss = (1 - cossim)
    if average:
        return loss.mean()
    else:
        return loss

def cossim(x, y, detach_y=True, average=True):
    assert x.ndim == 2
    assert y.ndim == 2
    if detach_y:
        y = y.detach()
    x = x / (torch.norm(x, 2, dim=-1, keepdim=True)+1e-5)
    y = y / (torch.norm(y, 2, dim=-1, keepdim=True)+1e-5)
    cossim = (x * y).sum(dim=-1)
    if average:
        return cossim.mean()
    else:
        return cossim
