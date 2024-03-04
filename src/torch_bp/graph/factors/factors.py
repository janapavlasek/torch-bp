import torch

from typing import Any
from numbers import Real
from torch_bp.util.distances import pairwise_euclidean_distance
from torch_bp.distributions import Distribution
from torch_bp.inference.kernels import Kernel
from torch_bp.distributions.distances import kernel_mmd


class UnaryFactor(object):
    """
    Defaul UnaryFactor class,
    implements a trivial factor that returns 0 for now
    """
    def __init__(self, alpha: Real = 1) -> None:
        self.alpha = alpha

    def __call__(self, x: Any) -> Any:
        return self.log_likelihood(x)

    def log_likelihood(self, x: Any) -> Any:
        print("WARNING: No log likelihood implemented. Returning zeros.")
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    def grad_log_likelihood(self, x: Any) -> Any:
        """
        Returns the grad_log_likelihood and log_likelihood,
        by default implements auto-gradient
        """
        x = x.detach().requires_grad_(True)
        unary = self.log_likelihood(x)
        grad_unary, = torch.autograd.grad(unary.sum(), x)  # TODO: consider use jacrev instead
        return self.alpha * grad_unary, self.alpha * unary


class DistributionUnaryFactor(UnaryFactor):
    """
    For cases where factor is equivalent to a known Distribution
    """
    def __init__(self, p_x: Distribution, alpha: Real = 1) -> None:
        super(DistributionUnaryFactor, self).__init__(alpha=alpha)

        self.p_x = p_x

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        return self.p_x.log_pdf(x)


class PairwiseFactor(object):
    def __init__(self, alpha: Real = 1) -> None:
        self.alpha = alpha

    def __call__(self, x_s: Any, x_t: Any) -> Any:
        return self.log_likelihood(x_s, x_t)

    def log_likelihood(self, x_s: Any, x_t: Any) -> Any:
        raise NotImplementedError()


class DistancePairwiseFactor(PairwiseFactor):
    """
    Factor implements energy function that increases as the norm2 distance between
    x_s and x_t deviates from expected `dist`

    E = -(dist - ||x_s - x_t||_2)**2
    """
    def __init__(self, dist: torch.Tensor, alpha: Real = 1) -> None:
        super(DistancePairwiseFactor, self).__init__(alpha=alpha)
        self.dist = dist

    def log_likelihood(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        dists = torch.sqrt(pairwise_euclidean_distance(x_s, x_t))
        return -self.alpha * (dists - self.dist)**2


class KernelMMDFactor(PairwiseFactor):
    """
    Factor where the log_likelihood is equvilant to the MMD between x_s and x_t

    E = -MMD
    """
    def __init__(self, tf: torch.Tensor, kernel: Kernel, alpha: Real=100) -> None:
        super(KernelMMDFactor, self).__init__(alpha=alpha)

        self.tf = tf
        self.kernel = kernel

    def log_likelihood(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        mmd = kernel_mmd(x_s @ self.tf.T, x_t, self.kernel)
        return -mmd
