"""
Linearized Gaussian factors to be used together with Linear Gaussian BP
"""

import torch
from .factors import UnaryFactor, PairwiseFactor
from typing import Callable, Tuple, Union


class LinearizedGaussianEnergy(object):
    """
    Factor that generates a Gaussian Energy function.
    - Given a function, h, which takes in x, a Gaussian, Nand also a bias, z, and its expected covariance, sigma,
        generate the energy function that encodes this relation
        (ie the resulting output is a Gaussian distribution about input domain x that when h is applied to the distribution,
        the output will be a Gaussian such that h(x) ~ N(bias, sigma)).
    - This is not to be confused with the function h which just relates x to a given output h(x)
        (ie h is NOT the actual factor, the actual factor is this function which is a cost function)
    - Members:
        - h_w_grads : function which evaluates (jac_h(x), h(x))
        - z: (x_out_dim,) tensor, bias or the mean we want h(x) to have
        - sigma: (x_out_dim,x_out_dim) tensor, the covariance we want h(x) to hav
            the smaller the value for each element, the tighter the constraints are
        - x0: (x_in_0_dim+...x_in_N_dim), CURRENT linearization point (stacked)
        - energy_eta : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor, stored eta given CURRET linearization point
        - energy_lambda : (x_in_0_dim+x_in_1_dim+... x_in_N_dim, x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor,
            stored lambda given CURRENT linearization point
    - NOTE:
        - x_in_0_dim, ... x_in_N_dim -> dimensions of input to h
        - x_out_dim -> dimensions of output of h (NOT output of this cost)
        - since h is assumed to be linear, the distribution does not change wrt input x,
            however in actuality the h may not actually be linear, hence we have to relinearize about the new
            operating point each time x changes
    """
    def __init__(self, grads_w_h : Callable,
                 z : torch.Tensor, sigma : torch.Tensor,
                 x_0 : torch.Tensor,
                 linear : bool) -> None:
        """
        Inputs:
        - h : Callable, function which evaluates h(x)
        - h_w_grads : Callable, function which evaluates (jac_h(x), h(x))
        - z : (x_out_dim,) tensor, bias or the mean we want h(x) to have
        - sigma : (x_out_dim, x_out_dim) tensor, covariance we want h(x) to have
        - linear : bool, whether h(x) is linear
        """
        self._grads_w_h = grads_w_h
        self._z = z
        self._lambda = torch.linalg.inv(sigma)
        if linear:
            self._x_0 = x_0
            self._energy_eta, self._energy_lambda = self._linearize_energy_fn(x_0)
        self._call_fn = self._linear_call if linear else self._non_linear_call
        self._update_internals = self._update_internals_linear if linear else self._update_internals_nonlinear

    def __call__(self, x_0 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given linearization point about an x value, calculate the resultant energy function
        - Inputs:
            - x_0 : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor, linearization point
        - Returns:
            - energy_eta : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
            - energy_lambda : (x_in_0_dim+x_in_1_dim+... x_in_N_dim, x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
        """
        return self._call_fn(x_0)

    def update_bias(self, new_z: torch.Tensor, new_sigma: Union[None, torch.Tensor]= None):
        """
        Updates the bias and sigma of the energy function
        """
        self._z = new_z
        if new_sigma is not None:
            self._lambda = torch.linalg.solve(new_sigma, torch.eye(new_sigma.shape[-1]).to(new_sigma))
        self._update_internals()

    def _linearize_energy_fn(self, x_0 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to linearizes the energy function about a new point
        - Inputs:
            - x_0 : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
        - Returns:
            - energy_eta : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
            - energy_lambda : (x_in_0_dim+x_in_1_dim+... x_in_N_dim, x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
        """
        jac_h_x0, h_x0 = self._grads_w_h(x_0)
        deviation = jac_h_x0 @ x_0 + self._z - h_x0
        energy_eta = jac_h_x0.transpose(-2,-1) @ self._lambda @ deviation
        energy_lambda = jac_h_x0.transpose(-2,-1) @ self._lambda @ jac_h_x0
        return energy_eta, energy_lambda

    def _linear_call(self, x_0 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function to call if h is linear
        - Inputs:
            - x_0 : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor, linearization point
        - Returns:
            - energy_eta : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
            - energy_lambda : (x_in_0_dim+x_in_1_dim+... x_in_N_dim, x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
        """
        return self._energy_eta.clone(), self._energy_lambda.clone()

    def _non_linear_call(self, x_0 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function to call if h is non-linear
        - Inputs:
            - x_0 : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor, linearization point
        - Returns:
            - energy_eta : (x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
            - energy_lambda : (x_in_0_dim+x_in_1_dim+... x_in_N_dim, x_in_0_dim+x_in_1_dim+... x_in_N_dim) tensor
        """
        return self._linearize_energy_fn(x_0)

    def _update_internals_linear(self) -> None:
        """
        Method to update internal stored values after update bias is called for linear functions
        """
        self._energy_eta, self._energy_lambda = self._linearize_energy_fn(self._x_0)

    def _update_internals_nonlinear(self) -> None:
        """
        Method to update internal stored values after update bias is called for linear functions
        """
        pass


class NaryGaussianLinearFactor(object):
    """
    Generic N-ary factor definition for factor to N nodes that follows factor call signatures
    """
    def __init__(self, grads_w_h : Callable,
                 z : torch.Tensor, sigma : torch.Tensor,
                 x_0 : torch.Tensor,
                 linear : bool,
                 alpha=1) -> None:
        """
        Inputs:
        - h : Callable, function which evaluates h(x)
        - h_w_grads : Callable, function which evaluates (jac_h(x), h(x))
        - z : (x_out_dim,) tensor, bias or the mean we want h(x) to have
        - sigma : (x_out_dim, x_out_dim) tensor, covariance we want h(x) to have
        - linear : bool, whether h(x) is linear
        - alpha : float, scaling of factor
        """
        self.energy_fn = LinearizedGaussianEnergy(grads_w_h, z, sigma, x_0, linear)
        self.alpha = alpha

    def __call__(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.log_likelihood(*args)

    def log_likelihood(self, *args):
        """
        Inputs:
        - args : ((eta tensor, lambda tensor), ...) all attached node Gaussians
        Returns:
        - eta : (x_dim,) tensor
        - lambda : (x_dim,x_dim) tensor
        """
        means, _ = [x for x in zip(*args)]
        energy_eta, energy_lambda = self.energy_fn(torch.cat(means)) # set linearization point to be about mean
        return self.alpha * energy_eta, self.alpha * energy_lambda

    def update_bias(self, new_z: torch.Tensor, new_sigma: Union[None, torch.Tensor]= None):
        """
        Updates the bias and sigma of the energy function
        """
        self.energy_fn.update_bias(new_z, new_sigma)


class UnaryGaussianLinearFactor(UnaryFactor):
    """
    Wrapper to bridge NaryGaussianLinearFactor to UnaryFactor
    """
    def __init__(self, grads_w_h : Callable,
                 z : torch.Tensor, sigma : torch.Tensor,
                 x_0 : torch.Tensor,
                 linear : bool,
                 alpha=1) -> None:
        """
        Inputs:
        - h : Callable, function which evaluates h(x)
        - h_w_grads : Callable, function which evaluates (jac_h(x), h(x))
        - z : (x_out_dim,) tensor, bias or the mean we want h(x) to have
        - sigma : (x_out_dim, x_out_dim) tensor, covariance we want h(x) to have
        - linear : bool, whether h(x) is linear
        - alpha : float, scaling of factor
        """
        super().__init__(alpha)
        self.nary_factor = NaryGaussianLinearFactor(grads_w_h, z, sigma, x_0, linear)

    def log_likelihood(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
        - x : (eta tensor, lambda tensor)
        Returns:
        - eta : (x_dim,) tensor
        - lambda : (x_dim,x_dim) tensor
        """
        energy_eta, energy_lambda = self.nary_factor(x)
        return self.alpha * energy_eta, self.alpha * energy_lambda

    def update_bias(self, new_z: torch.Tensor, new_sigma: Union[None, torch.Tensor]= None):
        """
        Updates the bias and sigma of the energy function
        """
        self.nary_factor.update_bias(new_z, new_sigma)


class PairwiseGaussianLinearFactor(PairwiseFactor):
    """
    Wrapper to bridge NaryGaussianLinearFactor to PairwiseFactor
    """
    def __init__(self, grads_w_h : Callable,
                 z : torch.Tensor, sigma : torch.Tensor,
                 x_0 : torch.Tensor,
                 linear : bool,
                 alpha=1) -> None:
        """
        Inputs:
        - h : Callable, function which evaluates h(x)
        - h_w_grads : Callable, function which evaluates (jac_h(x), h(x))
        - z : (x_out_dim,) tensor, bias or the mean we want h(x) to have
        - sigma : (x_out_dim, x_out_dim) tensor, covariance we want h(x) to have
        - linear : bool, whether h(x) is linear
        - alpha : float, scaling of factor
        """
        super().__init__(alpha)
        self.nary_factor = NaryGaussianLinearFactor(grads_w_h, z, sigma, x_0, linear)

    def log_likelihood(self, x_s, x_t):
        """
        Inputs:
        - x_s : (eta tensor, lambda tensor)
        - x_t : (eta tensor, lambda tensor)
        Returns:
        - eta : (x_dim,) tensor
        - lambda : (x_dim,x_dim) tensor
        """
        energy_eta, energy_lambda = self.nary_factor(x_s, x_t)
        return self.alpha * energy_eta, self.alpha * energy_lambda

    def update_bias(self, new_z: torch.Tensor, new_sigma: Union[None, torch.Tensor]= None):
        """
        Updates the bias and sigma of the energy function
        """
        self.nary_factor.update_bias(new_z, new_sigma)
