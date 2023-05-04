import numpy as np
import cvxpy as cp
from utils import *
import time
from concurrent.futures import ThreadPoolExecutor, wait


class NonlinearBRSSplittingPoly:

    def __init__(self, train_set, val_set, state_limits, action_limits, target_set_lifted, L_lift,
                 x_dispersion, xu_dispersion, max_err_bound=0.2,
                 max_num_poly=32, max_n_rounds=3, n_threads=None):
        self.states_encoded, self.actions, self.next_states_encoded = train_set
        self.val_states_encoded, self.val_actions, self.val_next_states_encoded = val_set
        self.state_limits = state_limits
        self.action_limits = action_limits
        self.L_lift = L_lift

        self.max_err_bound = max_err_bound
        self.max_num_poly = max_num_poly
        self.max_n_rounds = max_n_rounds
        self.n_threads = n_threads
        self.x_dispersion = x_dispersion
        self.xu_dispersion = xu_dispersion
        self.setup()
        self.current_targets = [target_set_lifted]
        self.current_targets_parents = [None]
        self.targets_traj = [self.current_targets]
        self.targets_traj_nodes = []
        self.iter = 0

    def setup(self):
        self.A_global, self.B_global = cvx_opt_lr_elementwise(
            self.states_encoded, self.actions, self.next_states_encoded)
        print("Global A, B are set")

        L_err = []
        val_res = self.val_next_states_encoded - \
            self.val_states_encoded @ self.A_global.T - self.val_actions @ self.B_global.T
        valX = np.concatenate(
            [self.val_states_encoded, self.val_actions], axis=-1)
        for i,res_dim in enumerate(val_res.T):
            l = estimate_lipschitz_constant(valX, res_dim)
            while l is None:
                l = estimate_lipschitz_constant(valX, res_dim)
            L_err.append(l)
        self.L_err = np.array(L_err)

        print("Lipschitz constants for error:", self.L_err)
        print("Global dispersion x:{}, xu:{}".format(
            self.x_dispersion, self.xu_dispersion))
        print("Setup is completed\n")

    def _calibrate_single(self, target, states_encoded, actions, next_states_encoded, parent_node=None):
        # get the data in the target
        cond = points_in(target, next_states_encoded)
        if sum(cond) == 0:
            return None
        # get the domain
        domain = get_enclosing_oriented_box(
            states_encoded[cond, :self.nx], self.state_limits)
        if domain is None or domain.A.size == 0:
            return None
        # filter domain = domain \mink_sum x_dispersion
        filtr_domain = minkowski_sum_poly_box(domain, self.x_dispersion)
        cond2 = points_in(filtr_domain, states_encoded[:, :self.nx])
        states = states_encoded[cond2]
        actions = actions[cond2]
        next_states = next_states_encoded[cond2]
        try:
            # Adapt A,B and compute error bound
            A, B, error_bound_ct, error_bound_r = cvx_opt_constB_elementwise(states, actions, next_states,
                                                                             self.A_global, self.B_global,
                                                                             self.L_lift, self.L_err,
                                                                             x_dispersion=self.x_dispersion,
                                                                             xu_dispersion=self.xu_dispersion)
        except:
            return None
        return ReachNode(target, A, B, error_bound_ct, error_bound_r,
                         state_limits=domain,
                         parent_node=parent_node)

    def calibrate_single(self, target_set, parent=None):
        valid_splits = []
        initial_node = self._calibrate_single(
            target_set, *self.train_set, parent)
        if initial_node is None or initial_node.set_rep.A.size == 0:
            return []
        splits = [initial_node]
        i = 0
        while len(splits) > 0 and i < self.max_n_rounds:
            new_splits = []
            for split in splits:
                # if the error is feasible then use it as a valid split
                if np.sum(split.error_bound_r) < self.max_err_bound:
                    valid_splits.append(split)
                # if the error is infeasible then split
                else:
                    sp1, sp2 = [self._calibrate_single(h, *self.train_set, parent)
                                for h in polytope_split(split.set_rep, ratio=0.5,
                                                               over_dims=range(1, self.nx+1))]
                    # check if one of the splits are empty
                    if sp1 is not None and sp1.set_rep.A.size != 0:
                        new_splits.append(sp1)
                    if sp2 is not None and sp2.set_rep.A.size != 0:
                        new_splits.append(sp2)
            splits = new_splits
            i += 1
        # check the remaining splits
        for split in splits:
            if np.sum(split.error_bound_r) < self.max_err_bound:
                valid_splits.append(split)
        # filter empty ones
        valid_splits = [vs for vs in valid_splits if vs is not None]
        return valid_splits

    def calibrate_multi(self, targets, parents, verbose):
        def _fn(i):
            target, parent = targets[i], parents[i]
            calibrated_nodes = self.calibrate_single(target, parent)
            if parent is not None:
                parent.children.extend(calibrated_nodes)
            next_targets_nodes.extend(calibrated_nodes)
        if verbose:
            print("Calibrating polytopes")
            start = time.time()
        next_targets_nodes = []
        # do threading if requested
        if self.n_threads is not None:
            tasks = []
            n_threads = min(self.n_threads, len(targets))
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i in range(len(targets)):
                    tasks.append(executor.submit(_fn, i))
            wait(tasks)
        # do sequential computation
        else:
            for i in range(len(targets)):
                _fn(i)
        if verbose:
            print("Calibration took {:.2f} seconds".format(time.time()-start))
        return next_targets_nodes

    def compute_koopman_brs_single(self, target):
        # prepare for pre computation
        state_limits = target.state_limits
        XZ = lift_polytope(
            state_limits, target.A.shape[1]-state_limits.A.shape[1])
        action_limits = self.action_limits
        # shrink the target by the error bound
        shrinked = minkowski_diff_poly_box(
            target.set_rep, target.error_bound_r, target.error_bound_ct)
        # empty polytope
        if shrinked.A.size == 0:
            return None
        try:
            # apply the pre
            pre = tulip_pre_proj(shrinked, target.A, target.B,
                                 lims2hrep(action_limits), X=XZ)
        except Exception as e:
            return None
        return pre

    def compute_koopman_brs_multi(self, target_nodes, verbose=0):
        pres = []
        parents = []

        def _fn(i):
            target = target_nodes[i]
            try:
                pre = self.compute_koopman_brs_single(target)
            except cp.SolverError:
                return
            if pre is not None and pre.A.size != 0:
                pres.append(pre)
                parents.append(target)
        # do threading if requested
        if self.n_threads is not None:
            tasks = []
            n_threads = min(self.n_threads, len(target_nodes))
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i in range(len(target_nodes)):
                    tasks.append(executor.submit(_fn, i))
            wait(tasks)
        # do sequential computation
        else:
            for i in range(len(target_nodes)):
                _fn(i)

        if verbose:
            print("Number of valid pre's: ", len(pres))
        return pres, parents

    def single_iteration(self, verbose=0):
        start = time.time()
        # obtain the targets by splitting
        next_targets_nodes = self.calibrate_multi(
            self.current_targets, self.current_targets_parents, verbose)
        # apply pre to the targets
        pres, parents = self.compute_koopman_brs_multi(
            next_targets_nodes, verbose)

        # filter empty ones
        pres_new = []
        parents_new = []
        empty = 0
        cplx = 0
        for pre, parent in zip(pres, parents):
            if pre.A.size != 0 and np.sum(np.all(self.states_encoded@pre.A.T <= pre.b, axis=1)) != 0:
                # filter out too complex polytopes (too many facets)
                if pre.A.shape[0] > 200:
                    cplx += 1
                    continue
                pres_new.append(pre)
                parents_new.append(parent)
            else:
                empty += 1
        if verbose:
            print("Killed {}/{} of polytopes bcz empty: ".format(empty, len(pres)))
            print("Killed {}/{} of polytopes bcz complex: ".format(cplx, len(pres)))

        pres = pres_new
        parents = parents_new
        self.targets_traj_nodes.append(parents)
        self.targets_traj.append(pres)

        # sample polytopes for next iteration (if too many)
        if len(pres) > self.max_num_poly:
            if verbose:
                print(
                    "Sampling polytopes {} -> {}".format(len(pres), self.max_num_poly))
            indices = sample_polytopes_dist(pres, self.max_num_poly)
            pres = [pres[i] for i in indices]
            parents = [parents[i] for i in indices]

        # update the current targets
        self.current_targets = pres
        self.current_targets_parents = parents
        if verbose:
            print("It took: {:.2f} seconds".format(time.time()-start))
            print(" ")

    def run(self, max_iter=10, verbose=0):
        start = time.time()
        # apply single iteration by max_iter times
        for _ in range(max_iter):
            if verbose:
                print("Iteration: ", self.iter)
                print("Number of targets: ", len(self.current_targets))
            self.single_iteration(verbose=verbose)
            if len(self.current_targets) == 0:
                print("No more targets, finishing")
                break
            self.iter += 1
        if verbose:
            print("Total time: ", time.time()-start)

    @property
    def nx(self):
        return self.state_limits.shape[1]

    @property
    def train_set(self):
        return (self.states_encoded, self.actions, self.next_states_encoded)

    @property
    def all_sets(self):
        return [s for brs in self.targets_traj for s in brs]

    @property
    def all_sets_enumerated(self):
        return [(i, s) for i, brs in enumerate(self.targets_traj) for s in brs]
