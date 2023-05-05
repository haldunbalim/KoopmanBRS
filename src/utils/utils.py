import numpy as np
import cvxpy as cp
import random

def cvx_opt_lr_elementwise(states, actions, next_states):
    nx, nu = states.shape[-1], actions.shape[-1]
    As = []
    Bs = []
    for i in range(nx):
        A = cp.Variable((1, nx))
        B = cp.Variable((1, nu))

        pred = states@A.T + actions@B.T
        residual = next_states[:, i:i+1] - pred
        err_term = cp.max(cp.abs(residual), axis=0)
        prob = cp.Problem(cp.Minimize(cp.sum(err_term)))
        prob.solve(solver=cp.MOSEK)
        As.append(A.value)
        Bs.append(B.value)
    return np.concatenate(As, axis=0), np.concatenate(Bs, axis=0)

def cvx_opt_constB_elementwise(states, actions, next_states, A_global, B_global, L_lift, L_err, x_dispersion, xu_dispersion):
    nx, _ = B_global.shape
    As, err_us, err_ls, lip_terms_ls = [], [], [], []
    for i in range(nx):
        A = cp.Variable((1, nx))

        pred = states@A.T + actions@B_global[i:i+1, :].T
        residual = next_states[:, i:i+1] - pred
        err_u = cp.max(residual, axis=0)
        err_l = cp.min(residual, axis=0)
        err_term = (err_u - err_l)/2
        lip_term_A = cp.sum(cp.abs(A-A_global[i:i+1, :]) * L_lift[i]) * x_dispersion
        lip_term_err = L_err[i] * xu_dispersion
        lip_terms = lip_term_A + lip_term_err
        error_bound = err_term + lip_terms
        prob = cp.Problem(cp.Minimize(cp.sum(error_bound)))
        prob.solve(solver=cp.MOSEK)

        As.append(A.value)
        err_us.append(err_u.value)
        err_ls.append(err_l.value)
        lip_terms_ls.append(lip_terms.value)

    A = np.concatenate(As, axis=0)
    err_u, err_l = np.array(err_us), np.array(err_ls)
    error_bound_ct = ((err_u + err_l)/2)[:, 0]
    error_bound_r = ((err_u - err_l)/2)[:, 0] + np.array(lip_terms_ls)
    return A, B_global, error_bound_ct, error_bound_r

def sample_polytopes_dist(polytopes, n_poly):
    if len(polytopes) <= n_poly:
        return polytopes
    centers = np.array([p.chebXc for p in polytopes])
    chosen = []
    for _ in range(n_poly):
        centroid = np.mean(centers, axis=0)
        furthest = np.argmax(np.linalg.norm(centers-centroid, axis=1))
        centers = np.delete(centers, furthest, axis=0)
        chosen.append(furthest)
    return chosen

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)