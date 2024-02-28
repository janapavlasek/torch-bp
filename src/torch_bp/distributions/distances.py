import torch


def kernel_mmd(x, y, kernel):
    M, N = x.shape[0], y.shape[0]  # Only works if at least 2 dims
    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxx = (kxx.sum() - torch.trace(kxx)) / (M * (M - 1))
    kyy = (kyy.sum() - torch.trace(kyy)) / (N * (N - 1))
    kxy = kernel(x, y).sum() * 2 / (M * N)
    return kxx + kyy - kxy
