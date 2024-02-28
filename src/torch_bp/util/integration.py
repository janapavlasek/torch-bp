import torch


class Integrator(object):
    def __init__(self):
        pass

    def __call__(self, f, inputs=[]):
        return self.eval(f, inputs)

    def eval(self, f, inputs=[]):
        raise NotImplementedError()


class SumIntegrator(Integrator):
    """Simple integrator that sums over all function values. Suitable for
    discrete distributions."""

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def eval(self, f, inputs=[]):
        """Samples must be provided for this one."""
        f_vals = f(self.samples, *inputs)  # (N, ...)
        res = torch.logsumexp(f_vals, dim=0)
        return res


class MonteCarloIntegrator(Integrator):
    def __init__(self, sample_dist, num_samples=100):
        super().__init__()
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self._samples = None
        self._weights = None

    def set_samples(self, samples, weights=None):
        self._samples = samples.detach().clone()
        if weights is not None:
            self._weights = weights.detach().clone()
        else:
            N = torch.as_tensor(samples.size(0), device=samples.device, dtype=samples.dtype)
            self._weights = torch.full((samples.size(0),), -torch.log(N), device=samples.device, dtype=samples.dtype)

    def sample(self):
        self._samples = self.sample_dist.sample(self.num_samples)
        self._weights = self.sample_dist.log_pdf(self._samples)  # Weights for the samples.

    def samples(self):
        return self._samples.detach().clone()

    def eval(self, f, inputs=[]):
        samples, weights = self._samples, self._weights
        if samples is None:
            samples = self.sample_dist.sample(self.num_samples)
            weights = self.sample_dist.log_pdf(samples)
        if weights is None:
            weights = self.sample_dist.log_pdf(samples)
        N = torch.as_tensor(samples.size(0), device=samples.device, dtype=samples.dtype)
        f_vals = f(samples, *inputs)
        res = torch.logsumexp(f_vals - weights.unsqueeze(-1), dim=0) - torch.log(N)
        return res
