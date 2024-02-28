import torch
from typing import Iterable, Union, Type, Tuple
from torch.func import vmap, jacrev
from .tensor_bp import TensorBP
from torch_bp.inference.svgd import SVGD
from torch_bp.inference.kernels import Kernel, BatchKernel
from torch.nn.functional import log_softmax
from torch_bp.graph import MRFGraph


class SVBP(TensorBP):
    """
    Non-loopy version of particle BP
    """
    def __init__(self,
                 particles: torch.Tensor, graph: MRFGraph, kernel: Kernel,
                 optim_type: Type = torch.optim.SGD, optim_kwargs: dict = {"lr": 0.1},
                 nodes_to_solve=None,
                 tensor_kwargs={'device': 'cpu', 'dtype': torch.float32}) -> None:
        """
        Inputs:
        - particles: (N,K,dim) torch.Tensor, initial particle estimates,
            (N: num of nodes, K: num of particles, dim: dimension of variable)
        - graph: MRFGraph, BP graph to be solved
        - kernel: Kernel, kernel function for SVGD
        - optim_type: type, type(optimizer) of the optimizer to be used,
            during solve, the necessary optimizer would be instantiated
        - optim_kwargs: dict, optim hyperparameters
        - nodes_to_solve: None | int | list of int,
            ids of nodes to solve for, or None to solve all node.
        """
        super().__init__(graph, tensor_kwargs=tensor_kwargs)
        self.N, self.K, self.D = particles.shape
        self.nodes_to_solve = nodes_to_solve

        self.compute_msg = self._compute_msg_recursive
        self.init_particles = particles.detach().clone().to(**tensor_kwargs)
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs

        # Set up SVGD solver.
        if nodes_to_solve is None:
            self._svgd = SVGD(particles, self.log_beliefs, BatchKernel(kernel), grad_log_px=self.grad_log_beliefs)
        else:
            self._svgd = SVGD(particles[nodes_to_solve], self.log_beliefs, kernel,
                              grad_log_px=self.grad_log_belief)

        self.reset_msgs()

    def solve(self, num_iters: int,
              nodes_to_solve: Union[None, int, Iterable[int]] = None,
              iter_fn=None, return_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve non loopy particle BP, only 1 message passing per iteration is required.
        Still require multiple iterations due to imperfect representation of true distribution upon init.
        Eager solve since msg accumulation and gradient propagation have to be carried out.

        Inputs:
        - num_iters: int, number of iterations
        - solve_node: None | int | Iterable[int], ids of nodes user is interested in solving,
            if None solve all nodes TODO: suggest to handle as lazy solve or unify with other modules
        - iter_fn: Callable, function to call at each iteration for plotting or logging.
        - return_weights: If True, return the weights for the final particles (default: False).

        Returns:
        - particles: (N,K,dim) tensor, particles representing samples from the distribution
        - weights: (N,K) tensor, corresponding weights for each particles
        """
        # init on request
        optim = self.optim_type(self.optim_parameters(), **self.optim_kwargs)

        # message passing and gradient updates
        for it in range(num_iters):
            self.precompute_unary()
            self.precompute_pairwise()
            self.pass_messages(normalize=True, precompute=False)  # only 1 message passing iteration req
            optim.zero_grad()
            self.update()
            optim.step()

            if iter_fn is not None:
                iter_fn(self, it)

        particles = self.particles(nodes_to_solve)
        if not return_weights:
            return particles

        if nodes_to_solve is None:
            nodes_to_solve = [i for i in range(particles.shape[0])]
        weights = torch.stack([self.compute_belief_weights(i, recompute_msgs=False, recompute_factors=False)
                               for i in nodes_to_solve]) \
            if isinstance(nodes_to_solve, Iterable) \
            else self.compute_belief_weights(nodes_to_solve, recompute_msgs=False, recompute_factors=False)
        return particles, weights.detach()

    def update(self, compute_msgs: bool = True) -> None:
        """
        Updates the particles in the nodes

        Inputs:
        - compute_msg: bool, whether to evoke a `pass_message` operation
        """
        if compute_msgs:
            self.pass_messages(precompute=True)
        if self.nodes_to_solve is None:
            self._svgd.update()
        else:
            self._svgd.update([self.nodes_to_solve])

    def reset(self, x: Union[None, torch.Tensor]) -> None:
        """
        Resets the nodes managed by solver

        Input:
        - x: None | torch.Tensor, reset if None, else reset all particles as x
        """
        if self.nodes_to_solve is None:
            self._svgd.reset(x)
        else:
            self._svgd.reset(x[self.nodes_to_solve])
            self.init_particles = x.detach()

    def optim_parameters(self) -> Iterable[torch.Tensor]:
        """
        Fetches optimizer parameters (particles) for torch.optimizer object

        Returns:
        - params: list of torch.Tensor, tensors to be optimized
        """
        return [self._svgd.optim_parameters()]

    def particles(self, s: Union[None, int, Iterable[int]] = None) -> torch.Tensor:
        """
        Fetches currently stored particles

        Input:
        - s: None | int | list of int, ids of nodes to fetch from,
            if None, fetches all particles (up to those defined in nodes_to_solve)

        Returns:
        - particles: torch.Tensor, stored particles
        """
        if s is None:
            if self.nodes_to_solve is None:
                return self._svgd.particles()

            self.init_particles[self.nodes_to_solve] = self._svgd.particles()
            return self.init_particles.detach()
        if self.nodes_to_solve is None:
            return self._svgd.particles()[s, :, :]
        if self.nodes_to_solve == s:
            return self._svgd.particles()
        return self.init_particles[s].detach()

    def normalize_msgs(self) -> None:
        """
        Normalizes all the stored messages
        """
        norm_msgs = []
        for msgs_s in self.log_msgs:
            norm_msgs_s = []
            for msg in msgs_s:
                norm_msgs_s.append(log_softmax(msg, dim=-1))

            norm_msgs.append(norm_msgs_s)

        self.log_msgs = norm_msgs

    def precompute_pairwise(self) -> None:
        """
        Precompute all the pairwise functions and stores themn to avoid having to compute them every time
        """
        x = self.particles()
        x1 = x[[e[0] for e in self.graph.edges], :, :]
        x2 = x[[e[1] for e in self.graph.edges], :, :]

        # Backward pass.
        self.pairwise_grad = []
        self.pairwise = []

        # If there are no edges, no need to compute pairwise.
        if len(self.graph.edges) == 0:
            return

        if self.graph.same_factor:
            # If the factor is the same for all nodes, we can compute this all at once.
            if hasattr(self.graph.edge_factors, "grad_log_likelihood"):
                # assume that provided input is batched for now
                # dpair_1, dpair_2, self.pairwise = vmap(self.graph.edge_factors.grad_log_likelihood)(x1, x2)
                dpair_1, dpair_2, self.pairwise = self.graph.edge_factors.grad_log_likelihood(x1, x2)
            else:
                jac = vmap(vmap(jacrev(self.graph.edge_factors, argnums=1), in_dims=(None, 0)))
                dpair_1 = jac(x2, x1).split(1)
                dpair_2 = jac(x1, x2).split(1)
                # Forward pass.
                self.pairwise = vmap(self.graph.edge_factors)(x1, x2)
            self.pairwise_grad = [[dp1.squeeze(), dp2.squeeze()] for dp1, dp2 in zip(dpair_1, dpair_2)]

        else:
            for i, e in enumerate(self.graph.edges):
                e1, e2 = e
                edge_factor = self.graph.edge_factor(e1, e2)
                if hasattr(edge_factor, "grad_log_likelihood"):
                    # This function provides its own gradient.
                    dpair_1, dpair_2, pair_12 = edge_factor.grad_log_likelihood(x[e1], x[e2])
                    dpair_1, dpair_2 = dpair_1.view(self.K, self.K, -1), dpair_2.view(self.K, self.K, -1)
                else:
                    # This function does not have its own likelihood gradient.
                    jac_i = vmap(jacrev(edge_factor, argnums=1), in_dims=(None, 0))
                    dpair_1 = jac_i(x[e2], x[e1])
                    dpair_2 = jac_i(x[e1], x[e2])
                    pair_12 = self.graph.pairwise(x[e1], x[e2], e1, e2)

                self.pairwise.append(pair_12)
                self.pairwise_grad.append([dpair_1, dpair_2])

            self.pairwise = torch.stack(self.pairwise)

    def get_pairwise(self, s: int, t: int,
                     return_grad: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the pairwise evaluation between node s and node t,
        if return_grad is True, also returns the gradients

        Inputs:
        - s: int, node s id
        - t: int, node t id
        - return_grad: bool, if True returns gradients along with evaluated pairwise

        Returns:
        - eval: torch.Tensor, evaluated pairwise potential
        - grad: torch.Tensor, gradients of evaluated pairwise potential,
            only returned if return_grad is True
        """
        edge_idx = self.graph._edge_idx(s, t)
        pair_st = self.pairwise[edge_idx, :, :]
        e1, e2 = self.graph.edges[edge_idx]
        if e1 == t and e2 == s:
            # We have pair_ts and want pair_st.
            pair_st = pair_st.t()

        if return_grad:
            # Gradient w.r.t. the first argument.
            grad_idx = 0 if s == e1 else 1
            grad_pair_st = self.pairwise_grad[edge_idx][grad_idx]
            return pair_st, grad_pair_st

        return pair_st

    def precompute_unary(self) -> None:
        """
        Precomputes the unary factors and store them
        """
        self.unary_grad = []
        self.unary = []
        for s in range(self.graph.N):
            x_s = self.particles(s)
            grad_unary_s, unary_s = self.graph.unary_factor(s).grad_log_likelihood(x_s)

            self.unary_grad.append(grad_unary_s)
            self.unary.append(unary_s)

    def precompute_unary_single(self, s) -> None:
        """
        Precomputes the unary of node s and store it

        Inputs:
        - s: int, node s id
        """
        x_s = self.particles(s)
        grad_unary_s, unary_s = self.graph.unary_factor(s).grad_log_likelihood(x_s)
        self.unary_grad[s] = grad_unary_s
        self.unary[s] = unary_s

    def precompute_pairwise_single(self, s: int) -> None:
        """
        Precomputes the pairwise to node s and stores it,
        only valid if all pairwise factor on the graph are the same

        Input:
        - s: int, node s id
        """
        if not self.graph.same_factor:
            raise NotImplementedError("Single pairwise precompute not implemented for multi-factor case.")

        x = self.particles()
        edges = torch.as_tensor(self.graph.edges, dtype=int)
        # Check which edges have a connection to this node.
        s_edges = (s == edges).nonzero()[:, 0]
        x1 = x[edges[s_edges][:, 0], :, :]
        x2 = x[edges[s_edges][:, 1], :, :]
        dpair_1, dpair_2, pairwise_s = self.graph.edge_factors.grad_log_likelihood(x1, x2)
        self.pairwise[s_edges] = pairwise_s
        for i in range(s_edges.size(0)):
            self.pairwise_grad[s_edges[i]][0] = dpair_1[i].squeeze()
            self.pairwise_grad[s_edges[i]][1] = dpair_2[i].squeeze()

    def pass_messages(self, normalize: bool = False, precompute: bool = False) -> None:
        """
        Perform one message passing iteration for all nodes.

        Warning: If use_precomputed is True, this assumes that the current
        state of the precomputed messages is associated with the current state
        of the particles (i.e. the particles have not been updated since the
        last time the factors were precomputed.)

        Inputs:
        - normalize: bool, whether to normalize the messages after the updates. (default: False)
        - use_precomputed: bool, whether to use precomputed pairwise and unary values. (default: False)
        """
        if precompute:
            self.precompute_unary()
            self.precompute_pairwise()

        self.log_msgs = [[None for n in nbrs] for nbrs in self.graph.nbrs]
        for t in range(self.graph.N):
            for s in self.graph.get_nbrs(t):
                if self.get_msg(t, s) is None:
                    m_t_to_s = self.compute_msg(t, s, recompute_msgs=False,
                                                recompute_factors=False, store_msgs=True)
                    self.set_msg(t, s, m_t_to_s)

        if normalize:
            self.normalize_msgs()

    def _compute_msg_recursive(self, t: int, s: int,
                               recompute_msgs: bool = True, recompute_factors: bool = True,
                               store_msgs: bool = False) -> torch.Tensor:
        """
        Computes the message between node t and node s (recurses to crawl graph till leaf node)

        Inputs:
        - t: int, node t id
        - s: int, node s id
        - recompute_msgs: bool, if True always run message passing to compute message, else
            only run message passing if no prior stored messages
        - recompute_factors: bool, if True recompute the factors, else fetch from last stored value
        - store_msgs: bool, if True, update the stored message with the calculated value

        Returns:
        - eval_msg: torch.Tensor, final evaluated message between node t and node s
        """
        if recompute_factors:
            x_s = self.particles(s)
            x_t = self.particles(t)
            unary_t = self.graph.unary_factor(t)(x_t)
            pair_st = self.graph.pairwise(x_s, x_t, s, t)
        else:
            unary_t = self.unary[t]  # (N_t,)
            pair_st = self.get_pairwise(s, t)  # (N_s, N_d) where pair_st[i, j] = pair(x_s[i], x_t[j])

        m_t_to_s = pair_st + unary_t.unsqueeze(0)

        for u in self.graph.get_nbrs(t):
            # Exclude the node receiving the current message.
            if u == s:
                continue

            m_u_to_t = None
            if not recompute_msgs:
                m_u_to_t = self.get_msg(u, t)
            if m_u_to_t is None:
                m_u_to_t = self._compute_msg_recursive(u, t, recompute_msgs=recompute_msgs,
                                                       recompute_factors=recompute_factors, store_msgs=store_msgs)

            m_t_to_s += m_u_to_t.unsqueeze(0)

            if store_msgs:
                self.set_msg(u, t, m_u_to_t)

        m_t_to_s = torch.logsumexp(m_t_to_s, dim=-1) - torch.log(torch.as_tensor(self.K))
        return m_t_to_s

    def grad_log_message(self, t: int, s: int,
                         use_precomputed: bool = True) -> torch.Tensor:
        """
        Get the gradients of the log message between node t and node s

        Inputs:
        - t: int, node t id
        - s: int, node s id
        - use_precomputed: bool, if True use precomputed gradients

        Returns:
        - grad: torch.Tensor, gradients between node s and node t
        """
        unary_t = self.unary[t]  # (N_t,)
        # (N_s, N_d) where pair_st[i, j] = pair(x_s[i], x_t[j])
        pair_st, grad_pair_st = self.get_pairwise(s, t, return_grad=True)

        m_t_to_s = pair_st + unary_t.unsqueeze(0)

        for u in self.graph.get_nbrs(t):
            # Exclude the node receiving the current message.
            if u == s:
                continue

            m_u_to_t = self.get_msg(u, t)
            m_t_to_s += m_u_to_t.unsqueeze(0)

        # Numerical stability.
        m_t_to_s -= m_t_to_s.max(1)[0].unsqueeze(-1)
        # Compute numerator and denominator.
        num = grad_pair_st * torch.exp(m_t_to_s).unsqueeze(-1)  # (N_s, N_t, D)
        num = num.sum(1)  # (N_s, D)
        denom = torch.exp(m_t_to_s)  # (N_s, N_t)
        denom = denom.sum(1)  # (N_s,)

        grad_m_ts = num / denom.unsqueeze(-1)
        return grad_m_ts

    def grad_log_belief(self, x_s: torch.Tensor, s: int,
                        use_precomputed: bool = True) -> torch.Tensor:
        """
        Evaluates the gradient of the log belief of node s

        Inputs:
        - x_s: torch.Tensor, variable at node s
        - s: int, node s id
        - use_precomputed: bool, if True use stored gradients, else recomputes them

        Returns:
        - grads: torch.Tensor, gradient of log belief
        """
        # Needs to be cloned because of runtime issues if grad_log_message requires grad (but only sometimes...)
        grad_log_ps = self.unary_grad[s].clone()
        for t in self.graph.get_nbrs(s):
            grad_log_ps += self.grad_log_message(t, s, use_precomputed=use_precomputed)
        return grad_log_ps

    def grad_log_beliefs(self, x: torch.Tensor, use_precomputed: bool = True) -> torch.Tensor:
        """
        Evaluates the gradients of all nodes to solve

        Inputs:
        - x: torch.Tensor, variable of all the nodes
        - use_precomputed: bool, if True, uses stored values, else recomputes them

        Returns:
        - grads: torch.Tensor, gradient of log belief of all nodes
        """
        grads = []
        for s in range(self.N):
            grads.append(self.grad_log_belief(x[s, :, :], s, use_precomputed=use_precomputed))
        return torch.stack(grads)

    def compute_msgs(self, x_s: torch.Tensor, s: int, recompute_factors: bool = True) -> Iterable[torch.Tensor]:
        """
        Computes all the messages going to node s

        Inputs:
        - x_s: torch.Tensor, variable at node s
        - s: int, node s id
        - recompute_factors: bool, if True recomputes all the factors, else use precomputed factors

        Returns:
        - msgs: list of torch.Tensor, all messages going s
        """
        msgs_to_s = []
        for t in self.graph.get_nbrs(s):
            m_t_to_s = self._compute_msg_recursive(t, s, recompute_factors=recompute_factors)
            msgs_to_s.append(m_t_to_s)
        return msgs_to_s

    def log_belief(self, xs: torch.Tensor, s: int,
                   recompute_msgs: bool = True, recompute_factors: bool = True) -> torch.Tensor:
        """
        Computes the log belief at node s

        Inputs:
        - xs: torch.Tensor, variable at node s
        - s: int, node s id
        - recompute_msgs: bool, if True recomputes all the messages, else use currect stored messages
        - recompute_factors: bool, if True recomputes all factors, else use stored precomputed factors

        Returns:
        - log_belief: torch.Tensor, computed log belief at node s
        """
        unary = self.graph.unary_factor(s)(xs) if recompute_factors else self.unary[s]
        if recompute_msgs:
            msgs = self.compute_msgs(xs, s, recompute_factors=recompute_factors)
        else:
            msgs = self.get_incoming_msgs(s)
        # Make sure there are messages before stacking. Note: Can't modify
        # log_ps in place because of gradient computation.
        if len(msgs) == 0:
            return unary

        log_ps = unary + torch.stack(msgs).sum(0)
        return log_ps

    def log_beliefs(self, x: torch.Tensor,
                    recompute_msgs: bool = True, recompute_factors: bool = True) -> torch.Tensor:
        """
        Computes the log beliefs of all nodes to solve

        Inputs:
        - x: torch.Tensor, variable for all nodes
        - recompute_msgs: bool, if True recomputes all msgs, else use stored msgs
        - recompute_factors: bool, if True recomputes all factors, else use stored precomputed factors

        Returns:
        - log_beliefs: torch.Tensor, log belief of all nodes to solve
        """
        bels = []
        for s in range(self.N):
            bels.append(self.log_belief(x[s, :, :], s, recompute_msgs, recompute_factors))
        return torch.stack(bels)

    def compute_belief_weights(self, s: Union[None, int, Iterable[int]],
                               recompute_msgs: bool = True, recompute_factors: bool = True) -> torch.Tensor:
        """
        Computes the weights of each belief

        Inputs:
        - s: None | int | list of int, ids of nodes to compute weights for,
            if None, compute weights all particles (up to those defined in nodes_to_solve)
        - recompute_msgs: bool, if True recomputes all msgs, else use stored msgs
        - recompute_factors: bool, if True recomputes all factors, else use stored precomputed factors

        Returns:
        - weights: torch.Tensor, weights of all defined nodes
        """
        x_s = self.particles(s)
        log_ps = self.log_belief(x_s, s, recompute_msgs, recompute_factors)
        w = log_ps - log_ps.max()
        w = torch.exp(w) / torch.exp(w).sum()
        return w


class LoopySVBP(SVBP):
    """
    Loopy Stein Variational BP implementation
    """
    def __init__(self,
                 particles: torch.Tensor, graph: MRFGraph, kernel: Kernel,
                 msg_init_mode: str = "uniform", optimize: bool = False, **kwargs) -> None:
        """
        Inputs:
        - particles: (N,K,dim) torch.Tensor, initial particle estimates,
            (N: num of nodes, K: num of particles, dim: dimension of variable)
        - graph: MRFGraph, BP graph to be solved
        - kernel: Kernel, kernel function for SVGD
        - optim_type: type, type(optimizer) of the optimizer to be used,
            during solve, the necessary optimizer would be instantiated
        - optim_hyperparams: dict, optim hyperparameters
        - msg_init_mode: str, method used to initialize/reset messages,
            available options:
                "uniform" (uniformly across all particles),
                "pairwise" (sample accross each pairs)
        - nodes_to_solve: None | int | list of int,
            ids of nodes to solve for, or None to solve all node.
        - optimize: bool, if True attempts to optimize by compiling some of the message computation steps
            (default: False)
        """
        self.msg_init_mode = msg_init_mode
        super().__init__(particles, graph, kernel, **kwargs)

        self.pairwise = None
        self.unary = None
        self.compute_msgs_sync = torch.compile(self._compute_msgs_sync) if optimize else self._compute_msgs_sync

    def solve(self, num_iters: int, msg_pass_per_iter: int,
              nodes_to_solve: Union[None, int, Iterable[int]] = None,
              iter_fn=None, return_weights: bool = False) -> Tuple[torch.Tensor]:
        """
        Solve non loopy particle BP, only 1 message passing per iteration is required.
        Still require multiple iterations due to imperfect representation of true distribution upon init.
        Eager solve since msg accumulation and gradient propagation have to be carried out.

        Inputs:
        - num_iters: int, number of iterations
        - msg_pass_per_iter: int, number of message passing rounds per iteration before resampling
        - nodes_to_solve: None | int | Iterable[int], ids of nodes user is interested in solving,
            if None solve all nodes
        - iter_fn: Callable, function to call at each iteration for plotting or logging.
        - return_weights: If True, return the weights for the final particles (default: False).

        Returns:
        - particles: (N,K,dim) tensor, particles representing samples from the distribution
        - weights: (N,K) tensor, corresponding weights for each particles
        """
        # init on request
        optim = self.optim_type(self.optim_parameters(), **self.optim_kwargs)

        # message passing and gradient updates
        for it in range(num_iters):
            self.precompute_unary()
            self.precompute_pairwise()
            for _ in range(msg_pass_per_iter):
                self.pass_messages(normalize=True, precompute=False)
            optim.zero_grad()
            self.update()
            optim.step()

            if iter_fn is not None:
                iter_fn(self, it)

        # return output
        particles = self.particles(nodes_to_solve)
        if not return_weights:
            return particles

        if nodes_to_solve is None:
            nodes_to_solve = [i for i in range(particles.shape[0])]
        weights = torch.stack([self.compute_belief_weights(i, recompute_factors=False)
                               for i in nodes_to_solve]) \
            if isinstance(nodes_to_solve, Iterable) \
            else self.compute_belief_weights(nodes_to_solve, recompute_factors=False)
        return particles, weights.detach()

    def reset_msgs(self) -> None:
        """
        Resets all stored messages using the defined msg_init_mode
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
                [torch.logsumexp(self.graph.pairwise(self.particles(s),
                                                     self.particles(t), s, t), dim=-1)
                 for t in self.graph.get_nbrs(s)]
                for s in range(self.graph.N)
            ]
        else:
            raise Exception(f"Unrecognized mode: {self.msg_init_mode}")

        self.pairwise = None
        self.unary = None

    def pass_messages(self, normalize: bool = False, precompute: bool = False) -> None:
        """
        Perform one message passing iteration for all nodes.

        Warning: If use_precomputed is True, this assumes that the current
        state of the precomputed messages is associated with the current state
        of the particles (i.e. the particles have not been updated since the
        last time the factors were precomputed.)

        Inputs:
        - normalize: bool, whether to normalize the messages after the updates. (default: False)
        - use_precomputed: bool, whether to use precomputed pairwise and unary values. (default: False)
        """
        # Ensure that the factors have been precomputed if we need them to be.
        if not precompute and (self.pairwise is None or self.unary is None):
            raise Exception("Error: Requested to use precomputed factors, but they have not been initialized.")

        if precompute:
            self.precompute_unary()
            self.precompute_pairwise()

        self.log_msgs = self.compute_msgs_sync(self.unary, self.pairwise, self.log_msgs)

        if normalize:
            self.normalize_msgs()

    def _compute_msgs_sync(self, unary: torch.Tensor, pairwise: torch.Tensor,
                           log_msgs: Iterable[Iterable[torch.Tensor]]) -> Iterable[Iterable[torch.Tensor]]:
        """
        Function to roll the factor and message calculation into one function for compilation

        Inputs:
        - unary: torch.Tensor, value of all evaluated unary factors in graph
        - pairwise: torch.Tensor, value of all evaluated pairwise factors in graph
        - log_msgs: list of list of torch.Tensor, list of log messages going to every node

        Returns:
        - new_msgs: list of list of torch.Tensor, list of new log messages going to every node
        """
        new_msgs = [[None for n in nbrs] for nbrs in self.graph.nbrs]
        for e_idx, e in enumerate(self.graph.edges):
            s, t = e
            # Compute msg_{t->s}
            s_idx = self.graph._nbr_idx(s, t)
            pair_st = pairwise[e_idx, ...]
            new_msgs[t][s_idx] = self._compute_msg_loopy(t, s, unary[t], unary[s], pair_st, log_msgs)

            # Compute msg_{s->t}
            t_idx = self.graph._nbr_idx(t, s)
            pair_ts = pair_st.t()
            new_msgs[s][t_idx] = self._compute_msg_loopy(s, t, unary[s], unary[t], pair_ts, log_msgs)

        return new_msgs

    def _compute_msg_loopy(self, t: int, s: int,
                           unary_t: torch.Tensor, unary_s: torch.Tensor,
                           pairwise_st: torch.Tensor,
                           log_msgs: Iterable[Iterable[torch.Tensor]]) -> torch.Tensor:
        """
        Computes the message between node t and node s

        Inputs:
        - t: int, node t id
        - s: int, node s id
        - unary_t: torch.Tensor, evaluated unary potential at node t
        - unary_s: torch.Tensor, evaluated unary potential at node s
        - pairwise_st: torch.Tensor, evaluated pairwise potential between node t and node s

        Returns:
        - msg: torch.Tensor, message between node t and node s
        """
        m_t_to_s = pairwise_st + unary_t.unsqueeze(0)

        # Add incoming log messages.
        for u in self.graph.get_nbrs(t):
            if u == s:
                continue
            m_u_to_t = log_msgs[u][self.graph._nbr_idx(t, u)]
            m_t_to_s += m_u_to_t.unsqueeze(0)
        # To find log(m_{t->s})
        m_t_to_s = torch.logsumexp(m_t_to_s, dim=-1) - torch.log(torch.as_tensor(self.K))
        return m_t_to_s

    def log_belief(self, xs: torch.Tensor, s: int, recompute_factors: bool = True) -> torch.Tensor:
        """
        Calculates the log belief of node s

        Inputs:
        - xs: torch.Tensor, variable at node s
        - s: int, node s id
        - recompute_factors: bool, if True recomputes all factors, else use stored precomputed factors
        """
        if recompute_factors:
            unary = self.graph.unary_factor(s)(xs)
        else:
            unary = self.unary[s]

        # At this point, we don't care if it's loopy because messages should be passed first.
        msgs = self.get_incoming_msgs(s)
        if len(msgs) == 0:
            # If there are no incoming messages, return only the unary.
            return unary
        log_ps = unary + torch.stack(msgs).sum(0)
        return log_ps

    def compute_belief_weights(self, s: int, recompute_factors: bool = True) -> torch.Tensor:
        """
        Compute the weights of the beliefs on node s

        Inputs:
        - s: int, node s id
        - recompute_factors: bool, if True recomputes all factors, else use stored precomputed factors

        Retunrs:
        - weights: torch.Tensor, weights of node s
        """
        log_ps = self.log_belief(self.particles(s), s, recompute_factors=recompute_factors)
        w = log_ps - log_ps.max()
        w = torch.exp(w) / torch.exp(w).sum()
        return w
