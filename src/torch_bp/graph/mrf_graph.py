import numpy as np
import torch
from typing import Iterable, Tuple, Union

from .factors import PairwiseFactor, UnaryFactor


class MRFGraph(object):
    """
    Markov Random Field graph object

    Graph class for managing a Markov Random Field to be used by some BP algorithms.
    By definition, only unary and pairwise factors can be defined in this class.
    """
    def __init__(self, num_nodes: int,
                 edges: Iterable[Tuple[int, int]],
                 edge_factors: Union[None, Iterable[PairwiseFactor]] = None,
                 unary_factors: Union[None, Iterable[UnaryFactor]] = None) -> None:
        """
        Inputs:
        - num_nodes: int, number of nodes in the graph
        - edges: list of (int, int) tuples, definition of edges in the graph by node id, order sensitive
        - edge_factors: None | list of PairwiseFactor, pairwise factor for each edge,
            same order as number of edges, if None, initializes a default PairwiseFactor for all edges
        - unary_factors: None | list of UnaryFactor, unary factor for each node, according to node id,
            if None, initializes a default UnaryFactor class that does nothing
        """
        self.N = num_nodes
        self.edges = edges

        self.same_factor = True
        if edge_factors is not None:
            if isinstance(edge_factors, Iterable):
                assert len(edges) == len(edge_factors), "The number of edge factors and edges must be the same."
                self.same_factor = False
        else:
            # By default, no information from the edge fctor.
            edge_factors = PairwiseFactor()

        self.edge_factors = edge_factors

        if unary_factors is None:
            unary_factors = [UnaryFactor() for _ in range(self.N)]
        self.unary_factors = unary_factors

        # It's convenient computationally to have a list of neighbours per node.
        self.nbrs = [[] for _ in range(self.N)]
        self._nbr_edge_map = [[] for _ in range(self.N)]
        for i, e in enumerate(self.edges):
            n1, n2 = e
            # Add nodes to each other's neighbours.
            self.nbrs[n1].append(n2)
            self.nbrs[n2].append(n1)

            # Keep track of which neighbours correspond to which edge index.
            self._nbr_edge_map[n1].append(i)
            self._nbr_edge_map[n2].append(i)

    def _nbr_idx(self, s, t) -> int:
        """
        Finds the index of node s in the nbrs of t.

        Input:
        - s: int, node s id
        - t: int, node t id
        Returns:
        - indx: int, index of neighbour in node s
        """
        s_idx, = np.nonzero(np.array(self.nbrs[t]) == s)
        return s_idx[0]

    def _edge_idx(self, s: int, t: int) -> int:
        """
        Finds the index of the pairwise factor in node s that links it to node t

        Input:
        - s: int, node s id
        - t: int, node t id
        Returns:
        - indx
        """
        return self._nbr_edge_map[s][self._nbr_idx(t, s)]

    def get_nbrs(self, s: int) -> Iterable[int]:
        """
        Gets list of index that are neighbours with node s

        Input:
        - s: int, node s id
        Return:
        - neighbours: list of int, list of neighbour ids
        """
        return self.nbrs[s]

    def edge_factor(self, s: int, t: int) -> PairwiseFactor:
        """
        Get edge factor from node s to node t

        Inputs:
        - s: int, node s id
        - t: int, node t id
        Returns:
        - pairwise_factor: PairwiseFactor connecting s to t
        """
        if not isinstance(self.edge_factors, Iterable):
            return self.edge_factors
        return self.edge_factors[self._edge_idx(s, t)]

    def unary_factor(self, s: int) -> UnaryFactor:
        """
        Gets the unary factor attached to node s

        Input:
        - s: int, node s id
        Returns:
        - unary_factor: UnaryFactor for node s
        """
        return self.unary_factors[s]

    def pairwise(self,
                 x_s: torch.Tensor, x_t: torch.Tensor,
                 s: int, t: int) -> torch.Tensor:
        """
        Evaluates the pairwise factor connecting node s to node t

        Inputs:
        - x_s: torch.Tensor, value of node s to evaluate pairwise factor on
        - x_t: torch.Tensor, value of node t to evaluate pairwise factor on
        - s: int, node s id
        - t: int, node t id
        Returns:
        - eval: torch.Tensor, evaluated value on pairwise factor
        """
        factor = self.edge_factor(s, t)
        return factor(x_s, x_t)

    def is_leaf(self, s: int) -> bool:
        """
        Checks if node s is a leaf node.

        Inputs:
        - s: int, node s id
        Returns:
        - is_leaf: bool, true if node s is a leaf node
        """
        return len(self.nbrs[s]) <= 1


def edge_n_factor_to_mrf_graph(num_nodes: int,
                               factor_edges: Iterable[Iterable[int]],
                               factors: Iterable[Union[UnaryFactor, PairwiseFactor]]):
    """
    Util function to converts a set of factor_edges and factors into MRFGraph object

    Input:
    - num_nodes: int, number of nodes in graph
    - factor_edges: Iterable of tuple[int], each represent a connection (each either len 1 or 2)
    - factors: Iterable of UnaryFactor | PairwiseFactor, corresponding Factor object to each edge
    """
    # create dict from lists
    factor_dict = {factor_edge: factor for factor_edge, factor in zip(factor_edges, factors)}

    # get edge and edge_factors
    edges, edge_factors = zip(*[(edge, factor) for edge, factor in factor_dict.items() if len(edge) > 1])

    # get unary_factors, create UnaryFactor if not defined for a given id
    unary_factors = [factor_dict[(i,)] if (i,) in factor_dict else UnaryFactor() for i in range(num_nodes)]

    return MRFGraph(num_nodes, edges=edges, edge_factors=edge_factors, unary_factors=unary_factors)
