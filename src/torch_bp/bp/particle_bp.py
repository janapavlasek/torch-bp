import torch
from numbers import Real
from typing import Iterable, Union

import torch_bp.distributions as dist
from torch_bp.graph import MRFGraph
from .tensor_bp import TensorBP
from torch_bp.util.integration import MonteCarloIntegrator
from torch_bp.inference.sampling import importance_sample


class ParticleBP(TensorBP):
    """
    Non-loopy version of particle BP.
    """
    def __init__(self,
                 init_particles: torch.Tensor, graph: MRFGraph,
                 sample_mode: str = "nodes",
                 num_int_samples: int = 100, lims: Union[None, torch.Tensor] = None,
                 tensor_kwargs={'device': 'cpu', 'dtype': torch.float32}) -> None:
        """
        Inputs:
        - init_particles: (N,K,dim) torch.Tensor, particles used to init node
            (N: node num, K: num of particles, dim: dimension of variable)
        - graph: MRFGraph, graph to solve
        - sample_mode: str, method for integrating the particles,
            available options: "nodes" (just sample from particles), "unary" (follow unary distribution),
                "uniform" (uniformly sample accross entire known xy lims)
        - lims: None | torch.Tensor, used if sampling mode is "uniform", defines the limits to uniformly
            sample accross
        - tensor_kwargs: dict, tensor keyword args
        """
        if sample_mode == "nodes":
            integrators = [MonteCarloIntegrator(None) for _ in range(graph.N)]
        elif sample_mode == "unary":
            integrators = [MonteCarloIntegrator(graph.unary_factor(s).p_x, num_samples=num_int_samples)
                           for s in range(graph.N)]
        elif sample_mode == "uniform":
            integrators = MonteCarloIntegrator(
                dist.Uniform(lims[0, :], lims[1, :], **tensor_kwargs), num_samples=num_int_samples)
        else:
            raise Exception(f"Unrecognized sample mode: {sample_mode}")
        self.sample_mode = sample_mode

        super().__init__(graph, integrators, tensor_kwargs)

        self.N, self.K, self.D = init_particles.shape
        self._particles = init_particles.detach().clone().to(**tensor_kwargs)

        self.reset_msgs()

    def solve(self, num_iters: int, jitter_sigma: Real,
              nodes_to_solve: Union[None, int, Iterable[int]] = None,
              iter_fn=None, return_weights: bool = False):
        """
        Solve non loopy particle BP, only 1 message passing per iteration is required.
        Still require multiple iterations due to imperfect representation of true distribution upon init.

        Inputs:
        - num_iters: int, number of iterations
        - jitter_sigma: Real, standard deviation of noise added to particles
        - nodes_to_solve: None | int | Iterable[int], ids of nodes user is interested in solving,
            if None solve all nodes
        - iter_fn: Callable, function to call at each iteration for plotting or logging.
        - return_weights: If True, return the weights for the final particles (default: False).
        Returns:
        - particles: (N,K,dim) tensor, particles representing samples from the distribution
        - weights: (N,K) tensor, corresponding weights for each particles, if return_weights is True.
        """
        # message passing and resampling
        for it in range(num_iters):
            self.jitter(sigma=jitter_sigma)
            self.pass_messages()
            self.resample_belief()

            if iter_fn is not None:
                iter_fn(self, it)

        # particles and weights
        particles = self.particles(nodes_to_solve)

        if return_weights:
            if nodes_to_solve is None:
                nodes_to_solve = [i for i in range(self.graph.N)]
            weights = torch.stack([self.compute_belief_weights(i)
                                   for i in nodes_to_solve]) \
                if isinstance(nodes_to_solve, Iterable) \
                else self.compute_belief_weights(nodes_to_solve)
            return particles, weights

        return particles

    def particles(self, s: Union[None, int, Iterable[int]] = None) -> torch.Tensor:
        """
        Fetches particles of the defined nodes

        Inputs:
        - s: None, int, list of int,
            if None fetch all nodes, else fetches particles whose node id is defined in s
        Returns:
        - particles: torch.Tensor, requested particles
        """
        if s is None:
            return self._particles.detach().clone()

        return self._particles[s, :, :].detach().clone()

    def normalize_msgs(self) -> None:
        """
        Normalizes stored messages
        """
        norm_msgs = []
        for msgs_s in self.log_msgs:
            norm_msgs_s = []
            for msg in msgs_s:
                w = msg - msg.max()
                w = torch.exp(w) / torch.exp(w).sum()
                norm_msgs_s.append(torch.log(w))

            norm_msgs.append(norm_msgs_s)

        self.log_msgs = norm_msgs

    def pass_messages(self) -> None:
        """
        Single message passing sequence
        """
        self.reset_msgs()

        for s in range(self.graph.N):
            if self.sample_mode == "nodes":
                self.integrator(s).set_samples(self._particles[s])
            else:
                self.integrator(s).sample()

        for t, t_nbrs in enumerate(self.graph.nbrs):
            # Update messages from t to its nbr, s.
            for s in t_nbrs:
                x_s = self._particles[s]
                m_t_to_s = self.compute_log_msg(x_s, t, s)
                # Save this message.
                self.set_msg(t, s, m_t_to_s)

    def compute_belief_weights(self, s: Union[None, int, Iterable[int]]) -> torch.Tensor:
        """
        Computes the belief weights and return them

        Inputs:
        - s: None, int, list of int,
            if None fetch all nodes, else calculate weights whose node id is defined in s
        Returns:
        - weights: torch.Tensor, request weights
        """
        return self.belief(self._particles[s], s)

    def resample_belief(self) -> None:
        """
        Resamples the belief using the stored integrator method
        """
        for s in range(self.graph.N):
            ws = self.compute_belief_weights(s)
            idx = importance_sample(ws, self.K)
            self._particles[s] = self._particles[s, idx, :]

    def jitter(self, s: int = None, sigma: Real = 0.01) -> None:
        """
        Jitters the stored particles with 0 mean noise of sigma variance

        Inputs:
        - sigma: Real, amount of perturbation to jitter particles
        """
        if s is not None:
            self._particles[s, :, :] += torch.normal(0, sigma, size=(self.K, self.D), **self.tensor_kwargs)
        else:
            self._particles += torch.normal(0, sigma, size=(self.N, self.K, self.D), **self.tensor_kwargs)

    def set_particles(self, particles: torch.Tensor, s: int) -> None:
        """
        Set particles for a given node s

        Inputs:
        - particles: torch.Tensor, particles to set into node s
        - s: int, node s id
        """
        self._particles[s, :, :] = particles.detach().clone().to(**self.tensor_kwargs)


class LoopyParticleBP(ParticleBP):
    """
    Loopy version of particle BP
    """
    def __init__(self,
                 init_particles: torch.Tensor, graph: MRFGraph,
                 msg_init_mode: str = "uniform",
                 tensor_kwargs={'device': 'cpu', 'dtype': torch.float32}) -> None:
        """
        Inputs:
        - init_particles: (N,K,dim) torch.Tensor, particles used to init node
            (N: node num, K: num of particles, dim: dimension of variable)
        - graph: MRFGraph, graph to solve
        - msg_init_mode: str, method used to init messages,
            available options:
                "uniform" (uniformly across all particles),
                "pairwise" (sample accross each pairs)
        - tensor_kwargs: dict, tensor keyword args
        """
        self.msg_init_mode = msg_init_mode
        super().__init__(init_particles, graph, tensor_kwargs=tensor_kwargs)

        self.pairwise = None
        self.unary = None

    def solve(self, num_iters: int, msg_pass_per_iter: int, jitter_sigma: Real,
              nodes_to_solve: Union[None, int, Iterable[int]] = None, iter_fn=None,
              return_weights: bool = False):
        """
        Solve loopy particle BP, requires definition on number of iterations required.

        Inputs:
        - num_iters: int, number of iterations
        - msg_pass_per_iter: int, number of message passing rounds per iteration before resampling
        - jitter_sigma: Real, strength of jitter of particles
        - nodes_to_solve: None | int | Iterable[int], ids of nodes user is interested in solving,
            if None solve all nodes
        - iter_fn: Callable, function to call at each iteration for plotting or logging.
        - return_weights: If True, return the weights for the final particles (default: False).
        Returns:
        - particles: (N,K,dim) tensor, particles representing samples from the distribution
        - weights: (N,K) tensor, corresponding weights for each particles, if return_weights is True.
        """
        # message passing and resampling
        for it in range(num_iters):
            self.jitter(sigma=jitter_sigma)
            # If this graph is not loopy, you only need to call pass_messages() once.
            for _ in range(msg_pass_per_iter):
                self.pass_messages(normalize=True)
            self.resample_belief()

            if iter_fn is not None:
                iter_fn(self, it)

        # Grab current particles.
        particles = self.particles(nodes_to_solve)

        # If requested, compute the weights.
        if return_weights:
            if nodes_to_solve is None:
                nodes_to_solve = [i for i in range(self.graph.N)]
            weights = torch.stack([self.compute_belief_weights(i) for i in nodes_to_solve]) \
                if isinstance(nodes_to_solve, Iterable) \
                else self.compute_belief_weights(nodes_to_solve)
            return particles, weights

        return particles

    def reset_msgs(self) -> None:
        """
        Resets all stored messages using msg_init_mode method
        """
        # log_msgs[s, t] is log(msg_{s -> t})
        if self.msg_init_mode == "uniform":
            self.log_msgs = [
                [torch.log(torch.ones(self.K, dtype=torch.float32) / self.K)
                 for n in self.graph.get_nbrs(node)]
                for node in range(self.graph.N)
            ]
        elif self.msg_init_mode == "pairwise":
            self.log_msgs = [
                [torch.logsumexp(self.graph.pairwise(self._particles[s],
                                                     self._particles[t], s, t), dim=-1)
                 for t in self.graph.get_nbrs(s)]
                for s in range(self.graph.N)
            ]
        else:
            raise Exception(f"Unrecognized mode: {self.msg_init_mode}")

        self.pairwise = None
        self.unary = None

    def precompute_pairwise(self) -> None:
        """
        Precomputes all pairwise functions to avoid having to compute them every time.
        """
        self.pairwise = [
            self.graph.pairwise(self._particles[e1], self._particles[e2], e1, e2)
            for e1, e2 in self.graph.edges
        ]  # pairwise[i] is the pairwise message for edge i.

    def precompute_unary(self) -> None:
        """
        Precomputes all unary functions to avoid having to compute them every time.
        """
        self.unary = [self.graph.unary_factor(s)(self._particles[s]) for s in range(self.N)]

    def get_pairwise(self, s: int, t: int) -> torch.Tensor:
        """
        Evaluates the pairwise factor between node s and node t

        Inputs:
        - s: int, node s id
        - t: int, node t id
        Returns:
        - pairwise_potentials: torch.Tensor, evaluated pairwise potentials for factor between s and t
        """
        edge_idx = self.graph._edge_idx(s, t)
        pairwise_st = self.pairwise[edge_idx]

        first, second = self.graph.edges[edge_idx]
        if first == t and second == s:
            # This is the edge going the other way, so transpose first.
            pairwise_st = pairwise_st.t()

        return pairwise_st

    def pass_messages(self, normalize=False, use_precomputed=False) -> None:
        """
        Perform one message passing iteration for all nodes.

        Warning: If use_precomputed is True, this assumes that the current
        state of the precomputed messages is associated with the current state
        of the particles (i.e. the particles have not been updated since the
        last time the factors were precomputed.)

        Inputs:
        - normalize: Whether to normalize the messages after the updates. (default: False)
        - use_precomputed: Whether to use precomputed pairwise and unary values. (default: False)
        """
        # Ensure that the factors have been precomputed if we need them to be.
        if use_precomputed and (self.pairwise is None or self.unary is None):
            raise Exception("Error: Requested to use precomputed factors, but they have not been initialized.")

        new_msgs = [[None for n in nbrs] for nbrs in self.graph.nbrs]
        for t, t_nbrs in enumerate(self.graph.nbrs):
            # Log unary for each particle in t:
            x_t = self._particles[t]
            unary_t = self.graph.unary_factor(t)(x_t) if not use_precomputed else self.unary[t]
            # Update messages from t to its nbr, s.
            for s_idx, s in enumerate(t_nbrs):
                x_s = self._particles[s]
                new_msgs[t][s_idx] = self._compute_msg_loopy(x_t, x_s, t, s, unary_t=unary_t,
                                                             use_precomputed=use_precomputed)

        self.log_msgs = new_msgs

        if normalize:
            self.normalize_msgs()

    def _compute_msg_loopy(self,
                           x_t: torch.Tensor, x_s: torch.Tensor,
                           t: int, s: int,
                           unary_t: Union[None, torch.Tensor] = None, use_precomputed=False) -> torch.Tensor:
        """
        Method for computing message for a loopy BP from node t to node s

        Inputs:
        - x_t: torch.Tensor, particles at node t
        - x_s: torch.Tensor, particles at node s
        - t: int, node t id
        - s: int, node s id
        - unary_t: None | torch.Tensor, evaluated unary potential at t, None if not known
        - use_precomputed: bool, whether to use precomputed function output or recalculate
        Returns:
        - msg: torch.Tensor, message from node t to node s
        """
        # Ensure that the factors have been precomputed if we need them to be.
        if use_precomputed and (self.pairwise is None or self.unary is None):
            raise Exception("Error: Requested to use precomputed factors, but they have not been initialized.")

        if unary_t is None:
            unary_t = self.graph.unary_factor(t)(x_t) if not use_precomputed else self.unary[t]

        pair_ts = self.graph.pairwise(x_s, x_t, s, t) if not use_precomputed else self.get_pairwise(s, t)
        m_t_to_s = pair_ts + unary_t.unsqueeze(0)

        # Add incoming log messages.
        for u in self.graph.get_nbrs(t):
            if u == s:
                continue
            m_u_to_t = self.get_msg(u, t)  # Get previous message, don't recurse.
            m_t_to_s += m_u_to_t.unsqueeze(0)
        # To find log(m_{t->s})
        m_t_to_s = torch.logsumexp(m_t_to_s, dim=-1) - torch.log(torch.as_tensor(self.K))
        return m_t_to_s

    def compute_msgs(self, x_s: torch.Tensor, s: int,
                     use_precomputed: bool = False) -> Iterable[torch.Tensor]:
        """
        Compute messages going to node s

        Inputs:
        - x_s: torch.Tensor, particles at x_s
        - s: int, node s id
        - use_precomputed: bool, whether to use precomputed factor evaluations or recompute
        Returns:
        - msg: Iterable[torch.Tensor], messages going to node s
        """
        msgs_to_s = []
        for t in self.graph.get_nbrs(s):
            x_t = self._particles[t]
            m_t_to_s = self._compute_msg_loopy(x_t, x_s, t, s, use_precomputed=use_precomputed)
            msgs_to_s.append(m_t_to_s)
        return msgs_to_s

    def log_belief(self, xs: torch.Tensor, s: int, use_precomputed: bool = False) -> torch.Tensor:
        """
        Calculates the log belief of node s

        Inputs:
        - xs: torch.Tensor, variable at node s
        - s: int, node s id
        - use_precomputed: whether to precompute factor evaluations or recompute
        Returns:
        - log_belief: torch.Tensor, returns the log belief at node s
        """
        if not use_precomputed:
            unary = self.graph.unary_factor(s)(xs)
        else:
            unary = self.unary[s]

        # At this point, we don't care if it's loopy because messages should be passed first.
        msgs = self.compute_msgs(xs, s, use_precomputed=use_precomputed)
        if len(msgs) == 0:
            # If there are no incoming messages, return only the unary.
            return unary
        log_ps = unary + torch.stack(msgs).sum(0)
        return log_ps

    def compute_belief_weights(self, s: int) -> torch.Tensor:
        """
        Computes the weights for the belief at node s

        Input:
        - s: int, node s id
        Returns:
        - weight: torch.Tensor, weight of particles at node s
        """
        log_ps = self.log_belief(self._particles[s], s)
        # w = log_ps - log_ps.max()
        # w = torch.exp(w) / torch.exp(w).sum(dim=-1)
        w = torch.nn.functional.softmax(log_ps, dim=-1)
        return w
