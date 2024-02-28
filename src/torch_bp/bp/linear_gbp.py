"""
Implements linearized gaussian belief propagation according to
Davison, Andrew J. and Joseph Ortiz. “FutureMapping 2: Gaussian Belief Propagation for Spatial AI.”
ArXiv abs/1910.14139 (2019): n. pag.
(https://api.semanticscholar.org/CorpusID:207757428?utm_source=wikipedia)
"""

import torch
from torch_bp.graph.factor_graph import FactorGraph, FactorCluster
from torch_bp.graph.factors.linear_gaussian_factors import NaryGaussianLinearFactor
from torch_bp.bp import BeliefPropagation
from typing import Iterable, Union, Tuple
from .bp import BeliefPropagation

import warnings


class LinearGaussianBP(BeliefPropagation):
    """
    Belief propagation method using Gaussian BP.
    Additional linear description due to the way factors are assumed to be linearized each time.
    Internally passes message using a factor graph meaning message passing will be split to 2 portions:
    factors to node, and nodes to factor

    NOTE:
    - Due to the extensive use of inversion, the highest precision of float64 is HIGHLY recommended.
    - Currently, inversion operation ALWAYS convert to float64 and downcast back to lower type after operation.
    - This has been tested to be NOT sufficient.
    - Considering removing auto upcasting and letting users deal with the operation, especially considering some
        hardware architectures may not be compatible with torch.float64 operations (meaning users need to solve
        the problem themselves)
    """
    def __init__(self,
                 node_means: torch.Tensor, node_covars: torch.Tensor,
                 factor_graph: FactorGraph,
                 tensor_kwargs= {'device': 'cpu', 'dtype': torch.float64}) -> None:
        """
        Inputs:
        - node_means : (N,D) tensor, mean of nodes
        - node_covariances : (DxD) or (N,DxD) tensor, covariances of nodes
        - factor_graph : FactorGraph, graph with defined node and factor clusters
        NOTE:
        - assume that node data sizes are the same
        """
        # input checks
        mean_batch_shape = node_means.shape[:-1]
        covar_batch_shape = node_covars.shape[:-2]
        assert node_means.shape[-1] == node_covars.shape[-1], "Node mean data size (...,x_dim) needs to match covar size (...,x_dim,x_dim)"
        assert node_covars.shape[-1] == node_covars.shape[-2], "Node covariance needs to be square (...,x_dim,x_dim)"
        if len(covar_batch_shape) > 0:
            assert mean_batch_shape == covar_batch_shape, "Node covariance batch shape does not match node mean shape"
        else:
            node_covars = node_covars.expand(*mean_batch_shape, *node_covars.shape)
        if ('dtype' not in tensor_kwargs) or (tensor_kwargs['dtype'] != torch.float64):
            warnings.warn("Defined 'dtype' is not torch.float64. Note that inaccuracies will accumulate from inversion operations!")
        # variable init
        super().__init__(tensor_kwargs)
        self.factor_graph = factor_graph
        self.node_means = node_means.to(**tensor_kwargs)
        self.node_covars = node_covars.to(**tensor_kwargs)
        self.tensor_kwargs = tensor_kwargs
        # create databases to replace the one in BP
        self._precomputed_factor_db = None
        self.msg_factor_to_node_db = {(factor_id, node_id) : None
                                      for factor_id, factor_cluster in self.factor_graph.factor_clusters.items()
                                      for node_id in factor_cluster.neighbours}

    def solve(self, num_iters: int,
              nodes_to_solve: Union[None,int,Iterable[int]] = None) -> torch.Tensor:
        """
        Solve non loopy linear gaussian BP, only 1 message passing iteration is required.

        Inputs:
        - num_iters: int, number of iterations
        - nodes_to_solve: None | int | Iterable[int], ids of nodes user is interested in solving,
            if None solve all nodes
        Returns:
        - mean: (N,dim) tensor, mean of the request nodes
        - covar: (N,dim,dim) tensor, covars of the request nodes
        """
        for _ in range(num_iters):
            # precompute rollouts and pass messages
            self.pass_messages()

            # update beliefs (needed to evaluate f(x) at next cycle)
            self.update_beliefs()

        # fetch values
        means = self.node_means if nodes_to_solve is None \
            else self.node_means[nodes_to_solve]
        covars = self.node_covars if nodes_to_solve is None \
            else self.node_covars[nodes_to_solve]
        return means, covars

    def reset_msgs(self):
        """
        Sets all stored message to None
        """
        for i in self.msg_factor_to_node_db:
            self.msg_factor_to_node_db[i] = None

    def pass_messages(self):
        """
        Calculation for each message is split into 3 parts:
        1. precompute factors (since factors are only linearized once per pass_message iteration)
        2. calculate messages from nodes to factors
        3. calculate messages from factors to nodes
        - NOTE:
            - current version is not the most computationally efficient since it does alot of repeated calculations
            - however it is guaranteed to converge to the unique solution
        - TODO:
            - make this more efficient by reusing previously calculated results
            - extend this to beyond unary and pairwise factors
            - current bp msg store only supports pairwise and unary factor edge definitions
        """
        # precompute all factors in graph first
        self._precompute_factors()
        # iterate through all node clusters to find msg to factors and save messages after all calculations are done
        self.msg_factor_to_node_db = {(factor_id, node_id): self._compute_msg_from_factor(factor_cluster, node_id)
                                      for node_id, node_cluster in self.factor_graph.node_clusters.items()
                                      for factor_id, factor_cluster in node_cluster.factor_clusters.items()}

    def update_beliefs(self, pass_messages= False) -> None:
        """
        Updates the belief of all nodes using passed message
        - Inputs:
            - pass_messages : bool, define if a round of message passing is required before updating belief
        """
        if pass_messages:
            self.pass_messages()

        node_etas = torch.zeros_like(self.node_means).double() # required for accurate inversion
        node_lams = torch.zeros_like(self.node_covars).double()
        for node_id, node_cluster in self.factor_graph.node_clusters.items():
            msg_etas, msg_lambdas = zip(*[self.msg_factor_to_node_db[factor_id, node_id]
                                          for factor_id in node_cluster.factor_clusters])
            node_etas[node_id] = torch.stack(msg_etas).sum(dim=0)
            node_lams[node_id] = torch.stack(msg_lambdas).sum(dim=0)

        self.node_covars = torch.linalg.solve(node_lams, torch.eye(node_lams.shape[-1]).to(node_lams))
        self.node_means = (self.node_covars @ node_etas[...,None])[...,0]

    def _precompute_factors(self) -> None:
        """
        Precomputes factors instead of calculating them during message passing.
        Stores precomputed factors in the database for quick access.
        """
        self._precomputed_factor_db = {factor_id : factor_cluster.factor(*[(self.node_means[i], self.node_covars[i])
                                                                          for i in factor_cluster.neighbours])
                                      for factor_id, factor_cluster in self.factor_graph.factor_clusters.items()}

    def _get_precomputed_factor(self, factor_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fetch precomputed factors
        """
        f_eta, f_lambda = self._precomputed_factor_db[factor_id]
        return f_eta.clone(), f_lambda.clone() # cloned to prevent unintentional edits to original

    def _compute_msg_from_factor(self, source_factor_cluster: FactorCluster, target_node_id: int
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to calculate the message from factor to a node,
        msg_{fk->}
        - Returns:
            - message eta : (x_dim) tensor, inverse mean of gaussian variable or zeros
            - message lambda : (x_dim,x_dim) tensor, information matrix of gaussian variable or zeros
        """
        # fetch precomputed factor
        f_eta, f_lambda = self._get_precomputed_factor(source_factor_cluster.id)
        # special case: target node is only node connected to source factor
        if len(source_factor_cluster.neighbours) == 1:
            return f_eta, f_lambda
        # get messages from all nodes connected to source factor except for target node using inner function
        # for target node assume msg are zeros (helps with shaping later)
        msgs = [self._inner_compute_msg_from_node(node_id, source_factor_cluster.id) if node_id != target_node_id
                else (torch.zeros_like(self.node_means[target_node_id]), torch.zeros_like(self.node_covars[target_node_id]))
                for node_id in source_factor_cluster.neighbours]
        # combine joint probabilities
        msg_etas, msg_lambdas = zip(*msgs)
        f_eta += torch.cat(msg_etas)
        f_lambda += torch.block_diag(*msg_lambdas)
        # marginalize
        # shape finding
        target_node_nbr_ind = source_factor_cluster.neighbours.index(target_node_id)
        front = sum([self.node_means[i].shape[-1] for i in range(target_node_nbr_ind)])
        middle = self.node_means[target_node_nbr_ind].shape[-1]
        back = sum(self.node_means[i].shape[-1] for i in range(target_node_nbr_ind+1, len(source_factor_cluster.neighbours)))
        x_dim = front + middle + back
        # remapping
        R_a = torch.eye(x_dim, **self.tensor_kwargs)[[i for i in range(front, front+middle)]]
        R_b = torch.eye(x_dim, **self.tensor_kwargs)[[i for i in range(front)] + [i for i in range(front+middle, x_dim)]]
        eta_a, eta_b = R_a @ f_eta, R_b @ f_eta
        lambda_aa, lambda_bb = R_a @ f_lambda @ R_a.T, R_b @ f_lambda @ R_b.T
        lambda_ab, lambda_ba = R_a @ f_lambda @ R_b.T, R_b @ f_lambda @ R_a.T
        # marginalization calculation
        msg_eta = eta_a - lambda_ab @ torch.linalg.solve(lambda_bb.double(), eta_b.double()).to(f_eta)
        msg_lambda = lambda_aa - lambda_ab @ torch.linalg.solve(lambda_bb.double(), lambda_ba.double()).to(f_lambda)

        return msg_eta, msg_lambda

    def _compute_msg_from_node(self, source_node_id: int, target_factor_id: int
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to calculate the message from node to factor,
        msg_{xk->j}(fj) : gaussian message
        - Input:
            - factor_graph : FactorGraph, used for extracing other connected factors and nodes
        - Returns:
            - message_eta : (x_dim,) tensor, inverse mean of gaussian variable or zeros
            - message_lambda : (x_dim,x_dim) tensor, information matrix of gaussian variable or zeros
        - Implements: msg_{xk->j}(fj) = aggregate_mult_{f for n(xk) \ fj}(msg_{f->k}(xk))
        """
        node_cluster = self.factor_graph.node_clusters[source_node_id]
        # special case: target factor is only factor connected to source node
        if len(node_cluster.factor_clusters) == 1:
            return torch.zeros_like(self.node_means[source_node_id]), torch.zeros_like(self.node_covars[source_node_id])
        # get messages from all factors connected to source node except for target factor using inner function
        msgs = [self._inner_compute_msg_from_factor(factor_cluster, source_node_id)
                for factor_id, factor_cluster in node_cluster.factor_clusters.items()
                if factor_id != target_factor_id]
        # gaussian product cannonical
        msg_eta, msg_lambda = [torch.stack(msg_i).sum(dim=0) for msg_i in zip(*msgs)]

        return msg_eta, msg_lambda

    def _inner_compute_msg_from_factor(self, source_factor_cluster: FactorCluster, target_node_id: int
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inner loop functions to calculate msg to nodes and factors.
        In this case call upper compute_msg_from_node which also calls compute_msg_from_factor, resulting in recursion
        - Inputs:
            - source_factor_cluster: FactorCluster, factor and node neighbours the message is flowing from
            - target_node_id: int, node id the message is flowing to
        """
        return self._compute_msg_from_factor(source_factor_cluster, target_node_id)

    def _inner_compute_msg_from_node(self, source_node_id: int, target_factor_id: int
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inner loop functions to calculate msg to factor and nodes.
        In this case call upper compute_msg_from_factor which also calls compute_msg_from_node, resulting in recursion
        - Inputs:
            - source_node_id: int, node id the message is flowing from
            - target_factor_id: int, factor id the message is flowing to
        """
        return self._compute_msg_from_node(source_node_id, target_factor_id)


class LoopyLinearGaussianBP(LinearGaussianBP):
    """
    Loopy version of LinearGaussianBP,
    assumes that messages already exists and improve solution using message passing
    """
    def __init__(self, node_means: torch.Tensor, node_covars: torch.Tensor, factor_graph: FactorGraph,
                 init_covar= 1e5,
                 tensor_kwargs= {'device': 'cpu', 'dtype': torch.float64}) -> None:
        """
        Inputs:
        - node_means : (N,D) tensor, mean of nodes
        - node_covariances : (DxD) or (N,DxD) tensor, covariances of nodes
        - factor_graph : FactorGraph, graph with defined node and factor clusters
        - init_covar : float, a high value used for resetting messages such that
            new_covar = init_covar * eye
        - tensor_kwargs : dict, tensor keyword args used to create new tensors
        NOTE:
        - assume that node data sizes are the same
        """
        super().__init__(node_means, node_covars, factor_graph, tensor_kwargs)
        self._inv_covar = 1 / init_covar # used to generate inv(init_covar * torch.eye(x_dim)) => 1/init_covar * torch.eye(x_dim)
        # create a temp message database for messages from nodes to factor
        self.msg_node_to_factor_db = {(node_id, factor_id) : None
                                      for node_id, node_cluster in self.factor_graph.node_clusters.items()
                                      for factor_id in node_cluster.factor_clusters}
        # init messages to a defined start value
        self.reset_msgs()

    def solve(self, num_iters: int, msg_pass_per_iter: int,
              nodes_to_solve: Union[None,int,Iterable[int]] = None) -> torch.Tensor:
        """
        Solve loopy linear gaussian BP, update linear factors once every iter (each iter containing
        msg_pass_per_iter message passing steps)

        Inputs:
        - num_iters: int, number of iterations
        - msg_pass_per_iter: int, number of message passing rounds per iteration before resampling
        - nodes_to_solve: None | int | Iterable[int], ids of nodes user is interested in solving,
            if None solve all nodes
        Returns:
        - mean: (N,dim) tensor, mean of the request nodes
        - covar: (N,dim,dim) tensor, covars of the request nodes
        """
        for _ in range(num_iters):
            for _ in range(msg_pass_per_iter):
                # ppass messages
                self.pass_messages()

            # update beliefs (needed to evaluate f(x) at next cycle)
            self.update_beliefs()

        # fetch values
        means = self.node_means if nodes_to_solve is None \
            else self.node_means[nodes_to_solve]
        covars = self.node_covars if nodes_to_solve is None \
            else self.node_covars[nodes_to_solve]
        return means, covars

    def reset_msgs(self):
        """
        Instead of setting messages to None, initialize them to a reasonable initial seed value.
        In this case assume they are infinitely large covariance gaussians about means of zeros,
        hence eta is still zero and lambda are inv(covaraince)
        """
        for factor_id, node_id in self.msg_factor_to_node_db:
            msg_dim = self.node_means[node_id].shape[-1]
            self.msg_factor_to_node_db[(factor_id, node_id)] = (torch.zeros(msg_dim, **self.tensor_kwargs),
                                                                self._inv_covar * torch.eye(msg_dim, **self.tensor_kwargs))
        for node_id, factor_id in self.msg_node_to_factor_db:
            msg_dim = self.node_means[node_id].shape[-1]
            self.msg_node_to_factor_db[(node_id, factor_id)] = (torch.zeros(msg_dim, **self.tensor_kwargs),
                                                                self._inv_covar * torch.eye(msg_dim, **self.tensor_kwargs))

    def pass_messages(self):
        """
        Instead of just iterating through messages going to nodes, we also iterate messages going to factors
        and store them.
        """
        # precompute all factors in graph first
        self._precompute_factors()
        # iterate through all factor cluster to find msg to nodes and save messages after all calculations are done
        self.msg_node_to_factor_db = {(node_id, factor_id): self._compute_msg_from_node(node_id, factor_id)
                                      for factor_id, factor_cluster in self.factor_graph.factor_clusters.items()
                                      for node_id in factor_cluster.neighbours}
        # iterate through all node clusters to find msg to factors and save messages after all calculations are done
        self.msg_factor_to_node_db = {(factor_id, node_id): self._compute_msg_from_factor(factor_cluster, node_id)
                                      for node_id, node_cluster in self.factor_graph.node_clusters.items()
                                      for factor_id, factor_cluster in node_cluster.factor_clusters.items()}

    def _inner_compute_msg_from_factor(self, source_factor_cluster: FactorCluster, target_node_id: int
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inner loop functions to calculate msg to nodes and factors.
        Instead of doing recurssive call, assume that the message already exist and pull that from database
        - Inputs:
            - source_factor_cluster: FactorCluster, factor and node neighbours the message is flowing from
            - target_node_id: int, node id the message is flowing to
        """
        return self.msg_factor_to_node_db[source_factor_cluster.id, target_node_id]

    def _inner_compute_msg_from_node(self, source_node_id: int, target_factor_id: int
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inner loop functions to calculate msg to factor and nodes.
        Instead of doing recurssive call, assume that the message already exist and pull that from database
        - Inputs:
            - source_node_id: int, node id the message is flowing from
            - target_factor_id: int, factor id the message is flowing to
        """
        return self.msg_node_to_factor_db[source_node_id, target_factor_id]