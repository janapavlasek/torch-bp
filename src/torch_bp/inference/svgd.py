import torch


class SVGD(object):
    def __init__(self, init_particles, log_density, kernel, alpha=1., grad_log_px=None):
        self.log_density = log_density
        self.kernel = kernel
        self.alpha = alpha
        self.grad_log_px = grad_log_px

        self._particles = init_particles.detach().clone().requires_grad_(True)

    def optim_parameters(self):
        return self._particles

    def particles(self):
        return self._particles.detach()

    def reset(self, particles):
        self._particles.data = particles.clone()

    def calc_weights(self, **kwargs):
        log_px = self.log_density(self._particles, **kwargs)
        w = log_px - log_px.max()
        w = torch.exp(w) / torch.exp(w).sum()
        return w.detach()

    def calc_grad_norm(self, mean=False):
        K = self._particles.size(-2)  # Particles should be of shape (..., K, D)
        grad_norm = (self._particles.grad * self._particles.grad).sum(-1)
        if mean:
            grad_norm = grad_norm.sum() / K
        return grad_norm.detach()

    def calc_grad_percentage(self, mean=False):
        grad_norm = (self._particles.grad * self._particles.grad).sum(-1)
        particle_norm = (self._particles * self._particles).sum(-1)
        percent = (grad_norm / particle_norm)
        if mean:
            percent = percent.mean()
        return percent.detach()

    def update(self, likelihood_args=[], create_graph=False):
        log_px, self._particles.grad = self.calc_grads(self._particles, likelihood_args=likelihood_args,
                                                       create_graph=create_graph)
        return log_px

    def grad_log_density(self, x, likelihood_args=[], create_graph=False):
        if self.grad_log_px is None:
            x = x.detach().requires_grad_(True)
            log_px = self.log_density(x, *likelihood_args)
            d_log_px, = torch.autograd.grad(log_px.sum(), x, create_graph=create_graph)
        else:
            log_px = None
            d_log_px = self.grad_log_px(x, *likelihood_args)
        return log_px, d_log_px

    def calc_grads(self, x, likelihood_args=[], create_graph=False):
        K = x.size(-2)  # Particles should be of shape (..., K, D)

        log_px, d_log_px = self.grad_log_density(x, likelihood_args=likelihood_args, create_graph=create_graph)
        k_xx, d_kxx, _ = self.kernel.backward(x, x)  # (..., K, K), (..., K, K, D)

        phi = (torch.matmul(k_xx, d_log_px) + self.alpha * d_kxx.sum(-3)) / K
        return log_px, -phi
