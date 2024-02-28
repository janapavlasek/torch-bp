import torch
from torch.func import jacrev


def pairwise_euclidean_distance(x, y, squared=True):
    """Returns squared Euclidean distance pairwise between x and y.

    If x is size (N, D) and y is size (M, D), returns a matrix of size (N, M)
    where element (i, j) is the distance between x[i, :] and y[j, :]. If x is
    size (N, D) and y is size (D,), returns a vector of length (N,) where each
    element is the distance from a row in x to y. If both have size (D,)
    returns a number.
    """
    if x.ndim == 1 and y.ndim == 1:
        # Shape is (D,), (D,).
        diffs = x - y
    else:
        diffs = (x.unsqueeze(1) - y).squeeze()
    dist = torch.sum(diffs**2, dim=-1)
    if not squared:
        dist = torch.sqrt(dist)
    return dist


def chamfer(p_samples, q_samples, reduction='mean'):
    # N_p, N_q = p_samples.size(0), q_samples.size(0)
    dists = pairwise_euclidean_distance(p_samples, q_samples, squared=True)

    if reduction == 'mean':
        cd = dists.min(0)[0].mean() + dists.min(1)[0].mean()
    elif reduction == 'sum':
        cd = dists.min(0)[0].sum() + dists.min(1)[0].sum()
    else:
        raise Exception("DISTANCES: Chamfer distance: Unrecognized reduction: {}".format(reduction))

    return cd


class PairwiseDistance(object):
    def __init__(self):
        pass

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplementedError()

    def backward(self, x, y):
        dist = self.forward(x, y)
        jac_fn = jacrev(self.forward)  # (N, D), (M, D) -> (N, M)
        d_x = jac_fn(x, y)             # (N, D), (M, D) -> (N, M, N, D).
        d_y = jac_fn(y, x)             # (M, D), (N, D) -> (M, N, M D).
        # Can sum along axis -2 since gradients are zero other than at (i, j, i, ...).
        return dist, d_x.sum(-2), d_y.sum(-2)


class SquaredEuclideanDistance(PairwiseDistance):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return pairwise_euclidean_distance(x, y)

    def backward(self, x, y):
        diffs = (x.unsqueeze(1) - y).squeeze()
        dist = torch.sum(diffs**2, dim=-1)

        return dist, 2 * diffs, -2 * diffs.transpose(0, 1)
