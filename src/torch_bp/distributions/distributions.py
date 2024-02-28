import torch
import torch.distributions as dist
import numpy as np

from torch_bp.util.distances import pairwise_euclidean_distance

INF = 1e6


class Distribution(object):
    def __init__(self, device="cpu", dtype=torch.float32):
        self.params = {"device": device, "dtype": dtype}

    def __call__(self, x):
        return self.pdf(x)

    def sample(self, N=1):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def log_pdf(self, x):
        return torch.log(self.pdf(x))

    def eval_grid(self, lims, n_samples=50):
        x = np.linspace(*lims[:2], n_samples)
        y = np.linspace(*lims[2:], n_samples)

        X, Y = np.meshgrid(x, y)
        var = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)

        Z = self.pdf(torch.tensor(var, **self.params)).reshape((n_samples, n_samples))

        return X, Y, Z.cpu().numpy()


class Gaussian(Distribution):
    def __init__(self, mu, sigma, device="cpu", dtype=torch.float32):
        super(Gaussian, self).__init__(device=device, dtype=dtype)

        # Mean and covariance.
        self.mu = mu.clone().detach() if torch.is_tensor(mu) else torch.tensor(mu)
        self.sigma = sigma.clone().detach() if torch.is_tensor(sigma) else torch.tensor(sigma)

        # Make sure everything is on the same device.
        self.mu = self.mu.to(**self.params)
        self.sigma = self.sigma.to(**self.params)

        self.mu = self.mu.view(-1)
        self.dim = self.mu.size(0)

        # Make sure sizes are correct.
        assert self.dim > 0, "Gaussian params must have at least one dimension. dim: {}".format(self.dim)

        # Make sure covariance is symmetric and positive semi-definite.
        self.sigma = self.sigma.view(self.dim, self.dim)
        assert torch.allclose(self.sigma, self.sigma.T), "Covariance should be symmetric."
        assert torch.all(torch.linalg.eigvals(self.sigma).real >= 0), "Covariance should be positive semi-definite."

        self._sigma_inv = torch.linalg.inv(self.sigma)
        self._sigma_det = torch.linalg.det(self.sigma)
        self._denom = torch.sqrt(self._sigma_det * (2 * torch.pi)**self.dim)

    def pdf(self, x):
        N = 1 if x.ndim == 1 else x.shape[0]
        x = x.view(N, -1)
        x = x - self.mu
        f = torch.exp(-0.5 * (torch.matmul(x, self._sigma_inv) * x).sum(-1)) / self._denom  # x.mm(self._sigma_inv)
        if N == 1:
            f = f.squeeze()
        return f

    def log_pdf(self, x):
        N = 1 if x.ndim == 1 else x.shape[0]
        x = x.view(N, -1)
        x = x - self.mu
        f = -0.5 * (torch.matmul(x, self._sigma_inv) * x).sum(-1) - torch.log(self._denom)  # x.mm(self._sigma_inv)
        return f

    def sample(self, N=1):
        p = dist.multivariate_normal.MultivariateNormal(self.mu, self.sigma)
        return p.sample((N,))


class Mixture(Distribution):
    def __init__(self, means, covs, weights=None, normalize=True,
                 device="cpu", dtype=torch.float32):
        super(Mixture, self).__init__(device=device, dtype=dtype)

        assert len(means) == len(covs)

        self.gaussians = [Gaussian(mu, sig, **self.params) for mu, sig in zip(means, covs)]
        self.K = len(means)
        self.dim = self.gaussians[0].dim
        # self.tensor_kwargs = tensor_kwargs

        if weights is not None:
            self.weights = torch.tensor(weights, **self.params)
            self.weights = torch.maximum(self.weights, torch.zeros_like(self.weights))
            if normalize:
                self.normalize()
        else:
            self.weights = torch.full((self.K,), 1. / self.K, dtype=torch.float)

    def means(self):
        return [g.mu for g in self.gaussians]

    def cov(self):
        return [g.sigma for g in self.gaussians]

    def normalize(self):
        self.weights = self.weights / torch.sum(self.weights)

    def pdf(self, x):
        N = 1 if x.ndim == 1 else x.shape[0]
        f = torch.zeros(N, **self.params)
        for g, w in zip(self.gaussians, self.weights):
            f += w * g.pdf(x)
        if N == 1:
            f.squeeze()
        return f

    def log_pdf(self, x):
        # Numerically stable log PDF.
        f = torch.stack([g.log_pdf(x) + torch.log(w) for g, w in zip(self.gaussians, self.weights)], dim=-1)
        return torch.logsumexp(f, dim=-1)

    def sample(self, N=1):
        idx = np.random.choice(np.arange(self.K), size=N, p=self.weights.numpy())
        x = [self.gaussians[i].sample().squeeze() for i in idx]
        return torch.stack(x)


class SampleMixture(Distribution):
    def __init__(self, samples, sigma, distance_fn=None, device="cpu", dtype=torch.float32):
        super(SampleMixture, self).__init__(device=device, dtype=dtype)

        self.samples = samples.detach().clone()
        self.K, self.D = self.samples.shape
        self.distance_fn = distance_fn if distance_fn is not None else pairwise_euclidean_distance

        self._error_px = Gaussian(0, sigma, **self.params)

    def pdf(self, x):
        dist = torch.sqrt(self.distance_fn(x, self.samples))
        px = self._error_px(dist.view(-1, 1)).view(-1, self.K).sum(-1) / self.K
        return px

    def log_pdf(self, x):
        dist = torch.sqrt(self.distance_fn(x, self.samples))
        log_pi_x = self._error_px.log_pdf(dist.view(-1, 1)).view(-1, self.K)
        log_px = torch.logsumexp(log_pi_x, dim=-1) - torch.log(torch.as_tensor(self.K, **self.params))
        return log_px


class Uniform(Distribution):
    def __init__(self, low, high, device="cpu", dtype=torch.float32):
        super(Uniform, self).__init__(device=device, dtype=dtype)

        self.low = low.clone().detach() if torch.is_tensor(low) else torch.tensor(low, **self.params)
        self.high = high.clone().detach() if torch.is_tensor(high) else torch.tensor(high, **self.params)

        assert self.low.ndim == 1 and self.high.ndim == 1, "Bounds must be one dimensional."
        assert self.low.size(0) == self.high.size(0), "Bounds must have the same shape."

        # self.params = {"device": device, "dtype": dtype}

        self.low = self.low.to(device=device, dtype=dtype)
        self.high = self.high.to(device=device, dtype=dtype)

        self.D, = self.low.shape
        self.max_pdf = 1. / torch.prod(torch.abs(self.high - self.low))
        self.max_log_pdf = -torch.log(torch.prod(torch.abs(self.high - self.low)))

    def is_inside(self, x):
        return torch.bitwise_and(torch.prod(x >= self.low, dim=-1), torch.prod(x <= self.high, dim=-1)).bool()

    def pdf(self, x):
        N = 1 if x.ndim == 1 else x.shape[0]
        px = torch.zeros(N, **self.params)
        px[self.is_inside(x)] = self.max_pdf
        return px

    def log_pdf(self, x):
        N = 1 if x.ndim == 1 else x.shape[0]
        px = torch.full((N,), -INF, **self.params)
        px[self.is_inside(x)] = self.max_log_pdf
        return px

    def sample(self, N=1):
        samples = [torch.FloatTensor(N).uniform_(self.low[i], self.high[i]) for i in range(self.D)]
        samples = torch.stack(samples, dim=-1).to(**self.params)
        return samples


class SmoothUniform(Uniform):
    def __init__(self, low, high, sigma=0.01, device="cpu", dtype=torch.float32):
        super(SmoothUniform, self).__init__(low, high, device=device, dtype=dtype)

        self._tails = Gaussian(torch.zeros_like(self.low), sigma * torch.eye(self.D, **self.params), **self.params)

    def log_pdf(self, x):
        x = (self.low - x).clamp(min=0) + (x - self.high).clamp(min=0)
        return self._tails.log_pdf(x)

    def pdf(self, x):
        x = (self.low - x).clamp(min=0) + (x - self.high).clamp(min=0)
        return self._tails(x)
