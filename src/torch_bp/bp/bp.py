from typing import Any
import torch


class BeliefPropagation(object):
    """
    Base class for all Belief Propagation (BP) solvers
    """
    def __init__(self,
                 tensor_kwargs={'device': 'cpu', 'dtype': torch.float32}) -> None:
        """
        Init parameters used for all BP solvers.

        Input:
        - tensor_kwargs: dict, tensor keyword args
        """
        self.tensor_kwargs = tensor_kwargs

    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """
        Solve the factor graph by running the BP algorithm.
        Lazy solve, belief is c.

        Input:
        - args: args, optional parameters to pass to lower level algorithm
        - kwargs: keyword args, optional named parameters to pass to lower level algorithm
        Returns:
        - ...: appropriate outputs for the solve function
        """
        raise NotImplementedError("`solve_function` in BPSolver have not been implemented.")
