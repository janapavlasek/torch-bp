from typing import Iterable, Union
import torch

from torch_bp.graph import MRFGraph
from .tensor_bp import TensorBP
from torch_bp.util.integration import SumIntegrator


class DiscreteBP(TensorBP):
    """
    Non-loopy version of particle BP.
    """

    def __init__(self,
                 node_vals: torch.Tensor, graph: MRFGraph,
                 tensor_kwargs={'device': 'cpu', 'dtype': torch.float32}) -> None:
        """
        Constructor for discrete BP.
        """
        # TODO: Allow nodes to have different values.
        self.node_vals = torch.as_tensor(node_vals).detach().clone().to(**tensor_kwargs)   # (K, D)

        super().__init__(graph, integrator=SumIntegrator(self.node_vals), tensor_kwargs=tensor_kwargs)

        self.reset_msgs()

    def solve(self, nodes_to_solve: Union[None,int,Iterable[int]] = None) -> torch.Tensor:
        """
        Solve non loopy discrete BP, only 1 message passing iteration is required.

        Inputs:
        - nodes_to_solve: None | int | Iterable[int], ids of nodes user is interested in solving,
            if None solve all nodes
        Returns:
        - belief: (K,) tensor, probability for each discrete value
        """
        self.pass_messages()
        return self.belief(nodes_to_solve)

    def pass_messages(self) -> None:
        """
        Evokes message passing routine.
        Non-loopy message routine, hence only 1 scan of the graph is required
        """
        self.reset_msgs()

        for t, t_nbrs in enumerate(self.graph.nbrs):
            # Update messages from t to its nbr, s.
            for s in t_nbrs:
                m_t_to_s = self.compute_log_msg(self.node_vals, t, s)
                # Save this message.
                self.set_msg(t, s, m_t_to_s)

    def log_belief(self, s: int) -> torch.Tensor:
        """
        Lazy fetching of log belief of node s
        Evokes calculation of the log belief from the messages
        """
        unary = self.graph.unary_factor(s)(self.node_vals)
        msgs = self.get_incoming_msgs(s)

        # Make sure there are messages before stacking. Note: Can't modify
        # log_ps in place because of gradient computation.
        if len(msgs) == 0:
            return unary

        log_ps = unary + torch.stack(msgs).sum(0)
        return log_ps

    def belief(self, s: Union[None, int, Iterable[int]]) -> torch.Tensor:
        """
        Lazy fetching of belief of all the nodes or selected nodes
        Calling this evokes calculation of belief from messages

        Input:
        - s: None | int | Iterable[int], node id(s) of nodes to query,
            if None return beliefs of all nodes

        Returns:
        - belief: tensor, belief of all the nodes, or selected nodes
        """
        if s is None:
            s = [i for i in range(self.graph.N)]
        log_bel = torch.stack([self.log_belief(si) for si in s]) \
            if isinstance(s, Iterable) else self.log_belief(s)
        # bel = log_bel - log_bel.max()
        # bel = torch.exp(bel) / torch.exp(bel).sum(dim=-1)
        bel = torch.nn.functional.softmax(log_bel, dim=-1)
        return bel
