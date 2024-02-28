import torch
import numpy as np

from torch.func import vmap, jacrev

from torch_bp.util.distances import SquaredEuclideanDistance


class Kernel(object):
    def __init__(self):
        pass

    def __call__(self, x, y, **kwargs):
        return self.forward(x, y, **kwargs)

    def forward(self, x, y):
        raise NotImplementedError()

    def backward(self, x, y):
        k_xy = self.forward(x, y)
        jac_fn = jacrev(self.forward)  # (N, D), (M, D) -> (N, M)
        d_kx, d_ky = jac_fn(x, y).sum(-2), jac_fn(y, x).sum(-2)
        return k_xy, d_kx, d_ky


class BatchKernel(Kernel):
    def __init__(self, kernel_fn):
        super(BatchKernel, self).__init__()
        self.kernel = kernel_fn

        self.forward = vmap(self.kernel)

    def backward(self, x, y):
        k_xy = self.forward(x, y)
        batch_jac_fn = vmap(jacrev(self.kernel))
        d_kx, d_ky = batch_jac_fn(x, y).sum(-2), batch_jac_fn(y, x).sum(-2)
        return k_xy, d_kx, d_ky


class RBFKernel(Kernel):
    def __init__(self, sigma=1, distance_fn=None):
        self.sigma = sigma  # Covariance.

        super(RBFKernel, self).__init__()

        if distance_fn is not None:
            self.distance_fn = distance_fn
        else:
            self.distance_fn = SquaredEuclideanDistance()

    def forward(self, x, y):
        dist = self.distance_fn(x, y)
        return torch.exp(-0.5 * dist / self.sigma)

    def backward(self, x, y):
        dist, d_diff_x, d_diff_y = self.distance_fn.backward(x, y)
        k_xy = torch.exp(-0.5 * dist / self.sigma)

        d_kx = -0.5 * d_diff_x * k_xy.unsqueeze(-1) / self.sigma
        d_ky = -0.5 * d_diff_y * k_xy.unsqueeze(-1) / self.sigma

        return k_xy, d_kx, d_ky


class RBFMedianKernel(RBFKernel):
    def __init__(self, sigma=None, gamma=None, distance_fn=None):
        super(RBFMedianKernel, self).__init__(distance_fn=distance_fn)
        self.gamma = None
        if gamma is not None:
            self.gamma = gamma
        elif sigma is not None:
            self.gamma = 1.0 / (2 * sigma ** 2)

    def forward(self, x, y, return_derivatives=False):
        dist = self.distance_fn(x, y)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.gamma is None:
            # h = np.median(dist.detach().cpu().numpy()) / (2 * np.log(x.size(0) + 1))
            # gamma = 1.0 / (1e-8 + 2 * h.item())
            gamma = self.median_heuristic(x.detach(), y.detach())
        else:
            gamma = self.gamma

        K_XY = (- gamma * dist).exp()

        if return_derivatives:
            dK_XY = torch.zeros_like(x)

            for i in range(x.shape[0]):
                dK_XY[i] = K_XY[i].matmul(x[i] - x) * 4 * gamma

            return K_XY, dK_XY

        else:
            return K_XY

    def backward(self, x, y):
        dist, d_diffs_x, d_diffs_y = self.distance_fn.backward(x, y)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.gamma is None:
            # h = np.median(dist.detach().cpu().numpy()) / (2 * np.log(x.size(0) + 1))
            # gamma = 1.0 / (1e-8 + 2 * h.item())
            gamma = self.median_heuristic(x.detach(), y.detach())
        else:
            gamma = self.gamma

        k_xy = (- gamma * dist).exp()

        d_kx = -2 * gamma * d_diffs_x * k_xy.unsqueeze(-1)
        d_ky = -2 * gamma * d_diffs_y * k_xy.unsqueeze(-1)

        return k_xy, d_kx, d_ky

    def set_params(self, x, y=None):
        self.gamma = self.median_heuristic(x, y)

    def median_heuristic(self, x, y=None):
        if y is not None:
            x = torch.concatenate([x, y], dim=0)
        N = x.size(0)
        dist = self.distance_fn(x, x).detach().cpu().numpy()
        h = np.median(dist[~np.eye(N, dtype=bool)])
        gamma = 1. / np.sqrt(0.5 * h)
        return gamma
