"Different loss functions to use."

from torch.nn.functional import mse_loss, smooth_l1_loss
import torch

def smoothness_loss(x):
    """Smoothness loss.
    This is the L1 norm of the second-order gradients for the predicted images.

    :param x: predicted image (NCHW) format."""
    def gradient(pred):
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy
    dx, dy = gradient(x)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    return torch.mean(torch.abs(dx2)[:, None]) + \
        torch.mean(torch.abs(dxdy)[:, None]) + \
        torch.mean(torch.abs(dydx)[:, None]) + \
        torch.mean(torch.abs(dy2)[:, None])
