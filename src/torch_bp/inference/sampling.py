import numpy as np
import torch

import torch_bp.distributions as dist


def normalize_log_weights(weights):
    """Numerically stable exponential."""
    w = weights - weights.max()
    weights = torch.exp(w) / torch.exp(w).sum()
    return weights


def importance_sample(weights, num_samples=1, keep_best=False):
    num_sample = num_samples if not keep_best else num_samples - 1
    indices = torch.multinomial(weights, num_sample, replacement=True)
    if keep_best:
        indices = torch.concat((indices, weights.argmax().view(1)))
    return indices


class Sampler(object):
    def __init__(self):
        pass

    def __call__(self, log_likelihood, num_samples=1, log_likelihood_args=[], *kwargs):
        return self.sample(log_likelihood, num_samples=num_samples,
                           log_likelihood_args=log_likelihood_args, *kwargs)

    def sample(self, log_likelihood, num_samples=1, log_likelihood_args=[], *kwargs):
        raise NotImplementedError()


class MCMC(Sampler):
    def __init__(self, dim=2, warmup=50, batch=50, candidate_cov=0.5, init_dist=None):
        super().__init__()
        self.warmup = warmup
        self.batch = batch
        self.candidate = dist.Gaussian(torch.zeros(dim), candidate_cov * torch.eye(dim))
        self.init_dist = init_dist if init_dist is not None else self.candidate

    def sample(self, log_likelihood, num_samples=1, log_likelihood_args=[]):
        samples = [self.init_dist.sample().view(-1)]
        fx = log_likelihood(samples[-1].view(1, -1), *log_likelihood_args)

        while len(samples) < (num_samples + self.warmup):
            prop_batch = samples[-1].view(1, -1) + self.candidate.sample(self.batch)
            f_prime = log_likelihood(prop_batch, *log_likelihood_args)

            r = np.random.random()
            accepted = r <= torch.exp(f_prime) / torch.exp(fx)
            if not torch.any(accepted):
                # If none of these samples are accepted, move on.
                continue

            # If one was good, grab the first one.
            prop = prop_batch[accepted.nonzero()[0]]
            fx = f_prime[accepted.nonzero()[0]]
            samples.append(prop.view(-1))

        return torch.stack(samples[self.warmup:])


class Gibbs(Sampler):
    def __init__(self, lims, num_candidates=50, warmup=10):
        super().__init__()
        self.lims = lims
        self.D = lims.size(0)
        self.num_candidates = num_candidates
        self.candidate = dist.Uniform(lims[0, :], lims[1, :])
        self.warmup = warmup

    def sample(self, log_likelihood, num_samples=1, log_likelihood_args=[]):
        current = self.candidate.sample().view(-1)
        samples = []

        for i in range(num_samples + self.warmup):
            candidates = self.candidate.sample(self.num_candidates)
            for d in range(0, self.D):
                other_idx = (torch.arange(self.D) != d).nonzero().squeeze()
                candidates_d = candidates.clone()
                candidates_d[:, other_idx] = current[other_idx]
                log_px = log_likelihood(candidates_d, *log_likelihood_args)
                idx = importance_sample(normalize_log_weights(log_px))
                current[d] = candidates_d[idx, d]
            samples.append(current.clone())

        return torch.stack(samples[self.warmup:])
