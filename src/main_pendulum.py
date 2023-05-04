from algo_split_poly import NonlinearBRSSplittingPoly
import numpy as np
import os
import pickle
from utils.geometry import lift_polytope, lims2hrep
from utils import set_seed
from simulation import PendulumModel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--lift-type", type=int, help="lifting type used", default=0)
args = parser.parse_args()

folder = "../outputs"
kwargs = {"max_num_poly": 32, "max_n_rounds": 3,
          "max_err_bound": 0.18, "n_threads": 8}
dyn_model = PendulumModel()
goal_psi_tol = 0.1
goal_psidot_tol = 0.5
state_limits = PendulumModel.state_limits
action_limits = PendulumModel.action_limits
set_seed(2)


class LiftingModel:
    def __init__(self, lift_type):
        self.lift_type = lift_type
        if self.lift_type == 0:
            self.L_lift = np.array([1, 1, 1])
        elif self.lift_type == 1:
            self.L_lift = np.array([1, 1])
        else:
            raise NotImplementedError("Unknown lift type", self.lift_type)


    def encode(self, x):
        th = x[..., :1]
        if self.lift_type == 0:
            return np.concatenate([x, np.sin(th)], axis=-1)
        elif self.lift_type == 1:
            return x


goal_state = np.array([goal_psi_tol, 0]).astype("f")
goal_tolerance = np.array(
    [goal_psi_tol, goal_psidot_tol])
target_set = lims2hrep(np.array([-goal_tolerance, goal_tolerance]), goal_state)


model = LiftingModel(args.lift_type)

with open(f"../data/pendulum/train_single.pkl", "rb") as f:
    data = pickle.load(f)
    train_states = model.encode(data["states"])
    train_next_states = model.encode(data["next_states"])
    train_actions = data["actions"].reshape(-1, 1)
train_set = (train_states, train_actions, train_next_states)

with open(f"../data/pendulum/val_single.pkl", "rb") as f:
    data = pickle.load(f)
    val_states = model.encode(data["states"])
    val_next_states = model.encode(data["next_states"])
    val_actions = data["actions"].reshape(-1, 1)
val_set = (val_states, val_actions, val_next_states)

# lift the target set
nx = state_limits.shape[-1]
n_lift = train_states.shape[-1] - nx
if n_lift > 0:
    target_set_lifted = lift_polytope(target_set, n_lift)
else:
    target_set_lifted = target_set

print(kwargs)
algo = NonlinearBRSSplittingPoly(train_set, val_set, state_limits, action_limits,
                                 target_set_lifted, L_lift=model.L_lift, x_dispersion=0.02, xu_dispersion=0.04,
                                 **kwargs)
print("Using", algo.__class__.__name__)
algo.run(15, verbose=1)

kwargs["algo"] = algo
kwargs["n_iter"] = len(algo.targets_traj)
os.makedirs(folder, exist_ok=True)
with open(f"{folder}/pendulum_brs_lifting={args.lift_type}.pkl", "wb") as f:
    pickle.dump(kwargs, f)
