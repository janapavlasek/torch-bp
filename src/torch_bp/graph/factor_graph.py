from typing import Iterable, Union

from .factors import UnaryFactor, PairwiseFactor
from .factors.linear_gaussian_factors import NaryGaussianLinearFactor


class FactorCluster(object):
    """
    A factor in a factor graph and the corresponding node indexes it is connected to
    """
    def __init__(self, id: int,
                 factor: Union[UnaryFactor, PairwiseFactor, NaryGaussianLinearFactor], neighbours: Iterable[int]) -> None:
        """
        Inputs:
        - id : int, a unique id for this particluar cluster (assigned by FactorGraph)
        - factor : factor object that links all the nodes in this cluster
        - neighbours : node id connected to this factor, as defined by user
        """
        self.id = id
        self.factor = factor
        self.neighbours = neighbours  # preserve order by user


class NodeCluster(object):
    """
    All factor clusters connected to the node index in a factor graph
    """
    def __init__(self, id: int, factor_clusters: Iterable[FactorCluster] = []) -> None:
        """
        Inputs:
        - id: int, node id
        - factor_clusters: list of FactorCluster, all factor clusters attached to this object
        """
        self.id = id
        self.factor_clusters = {factor_cluster.id : factor_cluster for factor_cluster in factor_clusters}


class FactorGraph(object):
    """
    A different representation of graph to aid with Linear Gaussian BP implementation.
    Factors are treated as edges on the graph and are not limited to unary and pairwise definition.
    """
    def __init__(self,
                 num_nodes: int,
                 factors: Iterable[Union[UnaryFactor, PairwiseFactor, NaryGaussianLinearFactor]],
                 factor_neighbours: Iterable[Iterable[int]]) -> None:
        """
        Input:
        - num_nodes : number of nodes in the graph, implcitly indexed from 0~(N-1)
        - factors : all factors
        - factor_neighbours : neighbour of each factor, represented as an iterable of node index
        """
        self.N = num_nodes
        self.factor_clusters = {id : FactorCluster(id, factor, nbrs)
                                for id, (factor, nbrs) in enumerate(zip(factors, factor_neighbours))}
        self.node_clusters = {id : NodeCluster(id) for id in range(num_nodes)}
        for factor_cluster_id, factor_cluster in self.factor_clusters.items():
            for node_id in factor_cluster.neighbours:
                self.node_clusters[node_id].factor_clusters[factor_cluster_id] = factor_cluster
