from typing import Iterable, Union
import torch
from .bp import BeliefPropagation
from torch_bp.graph import MRFGraph
from torch_bp.util.integration import Integrator


class TensorBP(BeliefPropagation):
    """
    Base class for all BP objects where the messages and beliefs are batched
    tensors and operations are implemented as PyTorch operations. Contains
    necessary information required by all related BP modules, such as graph and
    database of messages.
    Includes: DiscreteBP, ParticleBP, SVBP.
    """
    def __init__(self,
                 graph: MRFGraph, integrator: Union[Integrator, Iterable[Integrator]] = None,
                 tensor_kwargs={'device': 'cpu', 'dtype': torch.float32}) -> None:
        """
        Inputs:
        - graph: MRFGraph, graph used by BP algorithm
        - integrator: Integrator | list of Integrator, used by BP algorithm to integrate the distributions in
            each nodes, if only 1 integrator is defined, it is used for all nodes
        - tensor_kwargs: dict, tensor keyword args
        """
        super().__init__(tensor_kwargs)
        self.graph = graph
        self._integrator = integrator
        self.log_msgs = [
            [None for n in self.graph.get_nbrs(node)] for node in range(self.graph.N)
        ]  # msg[s, t] is log(msg_{s -> t})

    def integrator(self, s: int) -> Integrator:
        """
        Fetches integrator for node s

        Inputs:
        - s: int, node s id
        Returns:
        - integrator: Integrator, integrator for node s
        """
        if isinstance(self._integrator, Iterable):
            return self._integrator[s]
        return self._integrator

    def pass_messages(self) -> None:
        """
        Interface for passing pessages.
        To be implemented.
        """
        raise NotImplementedError()

    def reset_msgs(self) -> None:
        """
        Resets the stored messages.
        """
        self.log_msgs = [
            [None for n in self.graph.get_nbrs(node)] for node in range(self.graph.N)
        ]  # msg[s, t] is log(msg_{s -> t})

    def get_msg(self, t: int, s: int) -> torch.Tensor:
        """
        Get message from t->x, m_{t->s}

        Inputs:
        - t: int, node t id
        - s: int, node s id
        Returns:
        - message: torch.Tensor, for the set of known implementation return type is a tensor but might change
            for other novel types
        """
        # Get the index of neighbour s in t's nbrs.
        return self.log_msgs[t][self.graph._nbr_idx(s, t)]

    def set_msg(self, t: int, s: int,
                m_t_to_s: torch.Tensor) -> None:
        """
        Set message from t->s,  m_{t->s}

        Inputs:
        - t: int, node t id
        - s: int, node s id
        - m_t_to_s: torch.Tensor, message we are setting
        """
        # Get the index of neighbour s in t's nbrs.
        self.log_msgs[t][self.graph._nbr_idx(s, t)] = m_t_to_s

    def get_incoming_msgs(self, s: int) -> Iterable[torch.Tensor]:
        """
        Gets a list of all incoming messages to node s.

        Inputs:
        - s: int, node s id
        Returns:
        - msgs: list of torch.Tensor, messages going into node s
        """
        msgs = []
        for t in self.graph.get_nbrs(s):
            if t == s:
                continue
            msgs.append(self.get_msg(t, s))  # Add message from t->s
        return msgs

    def _msg_fn_recursive(self, x_t: torch.Tensor, x_s: torch.Tensor, t: int, s: int) -> torch.Tensor:
        """
        Compute message log(m_{t->s}), recursively.

        Inputs:
        - x_t: torch.Tensor, variable in node t
        - x_s: torch.Tensor, variable in node s
        - t: int, node t id
        - s: int, node s id
        Returns:
        - log_msg: torch.Tensor, log(msg) after all the iterative evaluation
        """
        N_t, N_s = x_t.size(0), x_s.size(0)
        unary_t = self.graph.unary_factor(t)(x_t)  # (N_t,)
        pair_ts = self.graph.pairwise(x_t, x_s, t, s).view(N_t, N_s)  # (N_t, N_s)
        m_t_to_s = pair_ts + unary_t.unsqueeze(-1)  # (N_t, N_s)

        # Add incoming log messages.
        for u in self.graph.get_nbrs(t):
            # Exclude the node receiving the current message.
            if u == s:
                continue

            # Recursive call to compute the message into this node.
            m_u_to_t = self.integrator(u)(self._msg_fn_recursive, [x_t, u, t])  # (N_t,)
            m_t_to_s += m_u_to_t.unsqueeze(-1)  # (N_t, N_s)

        return m_t_to_s

    def compute_log_msg(self, x_s: torch.Tensor, t: int, s: int) -> torch.Tensor:
        """
        Compute log m_{t->s}(x_s)

        Inputs:
        - x_s: torch.Tensor, variable at node s
        - t: int, node t id
        - s: int, node s id
        Returns:
        - log_msg: torch.Tensor, log(msg) evaluated using x_s
        """
        m_t_to_s = self.integrator(t)(self._msg_fn_recursive, [x_s, t, s])
        return m_t_to_s

    def log_belief(self, x_s: torch.Tensor, s: int, recompute: bool = True) -> torch.Tensor:
        """
        Computes the log belief of node s given its current belief x_s

        Inputs:
        - x_s: torch.Tensor, variable at node s
        - s: int, node s id
        - recompute: bool, whether to recompute the messages when calculating belief
        Returns
        - log_belief: torch.Tensor, evaluated log belief
        """
        log_ps = self.graph.unary_factor(s)(x_s)
        for t in self.graph.get_nbrs(s):
            if t == s:
                continue
            if recompute:
                msg_ts = self.compute_log_msg(x_s, t, s)
            else:
                msg_ts = self.get_msg(t, s)
            log_ps += msg_ts  # Add message from t->s

        return log_ps

    def belief(self, x_s: torch.Tensor, s: int) -> torch.Tensor:
        """
        Get Normalized belief of node s given its current belief

        Input:
        - x_s: torch.Tensor, variable at node s
        - s: int, node s id
        Returns:
        - belief: torch.Tensor, normalized belief at node s
        """
        log_bel = self.log_belief(x_s, s)
        bel = log_bel - log_bel.max()
        bel = torch.exp(bel) / torch.exp(bel).sum()
        return bel


class LoopyTensorBP(TensorBP):
    """
    Base class for all loopy BP objects where the messages and beliefs are
    batched tensors and operations are implemented as PyTorch operations.
    Contains necessary information required by all related BP modules, such as
    graph and database of messages. Differs from non loopy in that it assumes an
    initial set of messages and iteratively updates them, thus it does not crawl
    the graph to reach leaf nodes.
    In exchange, loses guarantees on exactness and requires multiple iterations
    to approximate messages.
    Includes: LoopyParticleBP, LoopySVBP.
    """
    def __init__(self,
                 graph: MRFGraph, integrator: Union[Integrator, Iterable[Integrator]] = None,
                 tensor_kwargs={'device': 'cpu', 'dtype': torch.float32}) -> None:
        """
        Inputs:
        - graph: MRFGraph, graph used by BP algorithm
        - integrator: Integrator | list of Integrator, used by BP algorithm to integrate the distributions in
            each nodes, if only 1 integrator is defined, it is used for all nodes,
        - tensor_kwargs= {'device': 'cpu', 'dtype': torch.float32}
        """
        super().__init__(graph, integrator=integrator, tensor_kwargs=tensor_kwargs)

    def reset_msgs(self) -> None:
        """
        Reset currently stored messages
        """
        self.log_msgs = [
            [torch.log(
                torch.ones(self.integrator(n).num_samples, **self.tensor_kwargs) / self.integrator(n).num_samples
             )
             for n in self.graph.get_nbrs(node)]
            for node in range(self.graph.N)
        ]

    def normalize_msgs(self) -> None:
        """
        Normalizes all currently stored messages
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

    def _msg_fn_loopy(self, x_t: torch.Tensor, x_s: torch.Tensor, t: int, s: int) -> torch.Tensor:
        """
        Compute message log(m_{t->s}), recursively (loopy version).

        Inputs:
        - x_s: torch.Tensor, variable at node s
        - x_t: torch.Tensor, variable at node t
        - t: int, node t id
        - s: int, node s id
        Returns:
        - msg: torch.Tensor, evaluated message after evaluating all the follow up calls
        """
        N_t, N_s = x_t.size(0), x_s.size(0)
        unary_t = self.graph.unary_factor(t)(x_t)  # (N_t,)
        pair_ts = self.graph.pairwise(x_t, x_s, t, s).view(N_t, N_s)  # (N_t, N_s)
        m_t_to_s = pair_ts + unary_t.unsqueeze(-1)  # (N_t, N_s)

        # Add incoming log messages.
        for u in self.graph.get_nbrs(t):
            # Exclude the node receiving the current message.
            if u == s:
                continue

            # Use last message.
            m_u_to_t = self.get_msg(u, t)
            m_t_to_s += m_u_to_t.unsqueeze(-1)  # (N_t, N_s)

        return m_t_to_s

    def compute_log_msg(self, x_s: torch.Tensor, t: int, s: int) -> torch.Tensor:
        """
        Compute log m_{t->s}(x_s)

        Inputs:
        - x_s: torch.Tensor, variable at node s
        - t: int, node t id
        - s: int, node s id
        Returns:
        - log_msg: torch.Tensor, evaluated log message from t to s
        """
        m_t_to_s = self.integrator(t)(self._msg_fn_loopy, [x_s, t, s])
        return m_t_to_s

    def pass_messages(self, iters: int, normalize: bool = True) -> None:
        """
        Perform N message passing iteration for all nodes.

        Warning: If use_precomputed is True, this assumes that the current
        state of the precomputed messages is associated with the current state
        of the particles (i.e. the particles have not been updated since the
        last time the factors were precomputed.)

        Inputs:
        - iters: int, number of message passing iterations
        - normalize: Whether to normalize the messages after the updates. (default: True)
        """
        for s in range(self.graph.N):
            self.integrator(s).sample()

        for it in range(iters):
            new_msgs = [[None for n in nbrs] for nbrs in self.graph.nbrs]
            for t, t_nbrs in enumerate(self.graph.nbrs):
                # Update messages from t to its nbr, s.
                for s_idx, s in enumerate(t_nbrs):
                    x_s = self.integrator(s).samples()
                    new_msgs[t][s_idx] = self.compute_log_msg(x_s, t, s)

            self.log_msgs = new_msgs

        if normalize:
            self.normalize_msgs()
