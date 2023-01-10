from abc import abstractmethod

import cvxopt
import cvxpy as cp
import numpy as np
import torch
from functions_evaluation import fastNonDominatedSort
from functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling
import functions_hv_grad_3d
from functions_evaluation import compute_hv_in_higher_dimensions as compute_hv
"""Implementation of Pareto HyperNetworks with:
1. Linear scalarization
3. EPO
EPO code from: https://github.com/dbmptr/EPOSearch
"""

class HvMaximization():
    """
    Mo optimizer for calculating dynamic weights using higamo style hv maximization
    based on Hao Wang et al.'s HIGA-MO
    uses non-dominated sorting to create multiple fronts, and maximize hypervolume of each
    """
    def __init__(self, n_mo_sol, n_mo_obj, ref_point, obj_space_normalize=True):
        super(HvMaximization, self).__init__()
        self.name = 'hv_maximization'
        self.ref_point = np.array(ref_point)
        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        self.obj_space_normalize = obj_space_normalize


    def compute_weights(self, mo_obj_val):
        n_mo_obj = self.n_mo_obj
        n_mo_sol = self.n_mo_sol

        # non-dom sorting to create multiple fronts
        hv_subfront_indices = fastNonDominatedSort(mo_obj_val)
        dyn_ref_point =  1.1 * np.max(mo_obj_val, axis=1)
        for i_obj in range(0,n_mo_obj):
            dyn_ref_point[i_obj] = np.maximum(self.ref_point[i_obj],dyn_ref_point[i_obj])
        number_of_fronts = np.max(hv_subfront_indices) + 1 # +1 because of 0 indexing
        
        obj_space_multifront_hv_gradient = np.zeros((n_mo_obj,n_mo_sol))
        # print(number_of_fronts)
        # print(n_mo_sol)
        for i_fronts in range(0,number_of_fronts):
            # compute HV gradients for current front
            temp_grad_array = grad_multi_sweep_with_duplicate_handling(mo_obj_val[:, (hv_subfront_indices == i_fronts) ],dyn_ref_point)
            #print(temp_grad_array)
            obj_space_multifront_hv_gradient[:, (hv_subfront_indices == i_fronts) ] = temp_grad_array

        # normalize the hv_gradient in obj space (||dHV/dY|| == 1)
        normalized_obj_space_multifront_hv_gradient = np.zeros((n_mo_obj,n_mo_sol))
        for i_mo_sol in range(0,n_mo_sol):
            w = np.sqrt(np.sum(obj_space_multifront_hv_gradient[:,i_mo_sol]**2.0))
            if np.isclose(w,0):
                w = 1
            if self.obj_space_normalize:
                normalized_obj_space_multifront_hv_gradient[:,i_mo_sol] = obj_space_multifront_hv_gradient[:,i_mo_sol]/w
            else:
                normalized_obj_space_multifront_hv_gradient[:,i_mo_sol] = obj_space_multifront_hv_gradient[:,i_mo_sol]
            #print(normalized_obj_space_multifront_hv_gradient)

        dynamic_weights = torch.tensor(normalized_obj_space_multifront_hv_gradient, dtype=torch.float)
        #print(dynamic_weights)
        return(dynamic_weights)

class MultiHead():
    
    def __init__(self, n_mo_sol, n_mo_obj, ref_point, obj_space_normalize=True):
        super().__init__()
        self.Hv_maximize = HvMaximization(n_mo_sol, n_mo_obj, ref_point)

    def get_weighted_loss(self,loss_numpy,device,loss_torch,head,penalty,lamda):
        n_samples, n_mo_obj, n_mo_sol = loss_numpy.shape
        dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
        #print(n_samples)
        for i_sample in range(0, n_samples):
          weights_task = self.Hv_maximize.compute_weights(loss_numpy[i_sample,:,:])
          dynamic_weights_per_sample[:, :, i_sample] = weights_task.permute(1,0)
        #print(dynamic_weights_per_sample)
        dynamic_weights_per_sample = dynamic_weights_per_sample.to(device)
        i_mo_sol = 0
        total_dynamic_loss = torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :]
                                                  * loss_torch[i_mo_sol], dim=0))
        for i_mo_sol in range(1, head):
          total_dynamic_loss += torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :] 
                                              * loss_torch[i_mo_sol], dim=0))
        for phat in penalty:
          total_dynamic_loss -= lamda*phat
        return total_dynamic_loss

class Solver:
    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks

    @abstractmethod
    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        pass

    def __call__(self, losses, ray, parameters, **kwargs):
        return self.get_weighted_loss(losses, ray, parameters, **kwargs)


class LinearScalarizationSolver(Solver):
    """For LS we use the preference ray to weigh the losses
    """

    def __init__(self, n_tasks):
        super().__init__(n_tasks)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        return (losses * ray).sum()


class EPOSolver(Solver):
    """Wrapper over EPO
    """

    def __init__(self, n_tasks, n_params):
        super().__init__(n_tasks)
        self.solver = EPO(n_tasks=n_tasks, n_params=n_params)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        assert parameters is not None
        return self.solver.get_weighted_loss(losses, ray, parameters)


class EPO:

    def __init__(self, n_tasks, n_params):
        self.n_tasks = n_tasks
        self.n_params = n_params

    def __call__(self, losses, ray, parameters):
        return self.get_weighted_loss(losses, ray, parameters)

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_weighted_loss(self, losses, ray, parameters):
        lp = ExactParetoLP(m=self.n_tasks, n=self.n_params, r=ray.cpu().numpy())

        grads = []
        for i, loss in enumerate(losses):
            g = torch.autograd.grad(loss, parameters, retain_graph=True)
            flat_grad = self._flattening(g)
            grads.append(flat_grad.data)

        G = torch.stack(grads)
        GG_T = G @ G.T
        GG_T = GG_T.detach().cpu().numpy()

        numpy_losses = losses.detach().cpu().numpy()

        try:
            alpha = lp.get_alpha(numpy_losses, G=GG_T, C=True)
        except Exception as excep:
            print(excep)
            alpha = None

        if alpha is None:  # A patch for the issue in cvxpy
            alpha = (ray / ray.sum()).cpu().numpy()

        alpha *= self.n_tasks
        alpha = torch.from_numpy(alpha).to(losses.device)

        weighted_loss = torch.sum(losses * alpha)
        return weighted_loss


class ExactParetoLP(object):
    """modifications of the code in https://github.com/dbmptr/EPOSearch
    """

    def __init__(self, m, n, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf     # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"

        return self.alpha.value


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        # return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a


class ChebyshevBasedSolver(Solver):
    """For Chebysev based solver, we use the preference ray to weigh the losses

    """
    def __init__(self, n_tasks):
        super().__init__(n_tasks)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        return torch.max(torch.abs(losses) * ray)


class UtilityBasedSolver(Solver):
    """For Utility based solver, we use the preference ray to weigh the losses

    """
    def __init__(self, n_tasks):
        super().__init__(n_tasks)
        self.upper_bound_loss = None

    def set_parameters(self, upper_bound_loss):
        self.upper_bound_loss = upper_bound_loss

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        # if losses.shape != self.upper_bound_loss.shape:
        #     raise Exception("Number of upper bound limit and objective not match")
        #print(losses)
        #print(ray)
        
        
        based_utilities = torch.pow(100 - losses, ray)

        #print(based_utilities)
        #print(1/torch.prod(based_utilities))
        return 1/torch.prod(based_utilities)