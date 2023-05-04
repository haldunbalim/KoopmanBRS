from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from utils.evt import evt_gev
import numpy as np
from itertools import product
from utils.geometry import lims2hrep, tulip_pre_proj, minkowski_diff_poly_box
import polytope as pc
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import cvx_opt_lr_elementwise
from utils import set_seed
from scipy.io import loadmat
from simulation import Simulator, DuffingModel
import os


nx = 2
Ns = 50
goal_tol = 0.1
num_tuples = 1250
set_seed(1)

states, actions, next_states = [], [], []
dyn_model = DuffingModel()
print(f"Collecting data...")
l, u = dyn_model.state_limits
for i in tqdm(range(num_tuples)):
    state = dyn_model.generate_random_state()
    action = dyn_model.generate_random_action()
    next_state = Simulator.successor_ivp(
        dyn_model, state, action, bound=False, integrate=True)
    states.append(np.array(state))
    actions.append(np.array(action))
    next_states.append(np.array(next_state))

states, actions, next_states = np.array(
    states), np.array(actions), np.array(next_states)
cutoff = int(0.8 * len(states))
train_states = states[:cutoff]
train_actions = actions[:cutoff]
train_next_states = next_states[:cutoff]
val_states = states[cutoff:]
val_actions = actions[cutoff:]
val_next_states = next_states[cutoff:]


class LiftingModel:
    def encode(self, states):
        x1 = states[..., :1]
        return np.concatenate([states, x1**3], axis=-1)


model = LiftingModel()

states_aug = model.encode(states)
next_states_aug = model.encode(next_states)
A, B = cvx_opt_lr_elementwise(states_aug, actions, next_states_aug)

val_states_aug = model.encode(val_states)
val_next_states_aug = model.encode(val_next_states)

pred = states_aug@A.T+actions@B.T
res = next_states_aug - pred

val_pred = val_states_aug@A.T+val_actions@B.T
val_res = val_next_states_aug-val_pred


# -------- EVT ---------
error_bound = []
for i, res_dim in enumerate(np.abs(val_res).T):
    e = evt_gev(res_dim, Ns=Ns)
    while e is None:
        e = evt_gev(res_dim, Ns=Ns)
    error_bound.append(e)
error_bound = np.array(error_bound)

# -------- Setup ---------
def lift_polytope(P, nd):
    return pc.Polytope(np.block([P.A, np.zeros((P.A.shape[0], nd))]), P.b)

goal_state = np.array([0, 0])

X = lims2hrep(dyn_model.state_limits)
XZ = lift_polytope(X, states_aug.shape[-1]-nx)
U = lims2hrep(dyn_model.action_limits)
W = lims2hrep(np.concatenate(
    [-error_bound[np.newaxis, :], error_bound[np.newaxis, :]]))

goal_lims = np.array([[-goal_tol]*2,
                      [goal_tol]*2])
target_set = lims2hrep(goal_lims)
target_set_lifted = lift_polytope(
    target_set, states_aug.shape[-1]-nx)

# -------- Pre ---------
start = time.time()
pres = [target_set_lifted]
for i in tqdm(range(10)):
    shrinked = minkowski_diff_poly_box(pres[-1], error_bound)
    if shrinked is None or shrinked.A.size == 0:
        break
    pre = tulip_pre_proj(shrinked, A, B, U, XZ)
    if pre is None or pre.A.size == 0:
        break
    pres.append(pre)
print("Finished")
print("It took: {:.3f} seconds".format(time.time()-start))


# -------- Visualization ---------
data = loadmat("../data/hjb_duffing.mat")["data"]
xspace = np.linspace(*dyn_model.state_limits[:, 0], data.shape[0])
yspace = np.linspace(*dyn_model.state_limits[:, 1], data.shape[1])
pts = np.array(list(product(xspace, yspace)))
dct = {}
for x in tqdm(xspace):
    for y in yspace:
        dct[(x, y)] = 0
        pt = model.encode(np.array([x, y]).astype("f"))
        contained = False
        for i, pre in enumerate(pres):
            if np.all(pre.A@pt <= pre.b):
                dct[(x, y)] = i + 1
                break
arr = np.array([[x, y, c] for (x, y), c in dct.items()])

hjb_color = "green"
our_color = "lightblue"
goal_color = "orange"
goal_tol = 0.1

fig = plt.figure()
ax = fig.add_subplot()
_pts = pts.reshape(data.shape[0], data.shape[1], 2)

ax.contour(_pts[:, :, 0], _pts[:, :, 1], data[:, :, -1]
           < 0, colors=[hjb_color], linewidths=[1])

ax.scatter(arr[arr[:, 2] > 0, 0], arr[arr[:, 2] > 0, 1], c=our_color, s=1)

ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.add_patch(Rectangle((-goal_tol, -goal_tol),
                       goal_tol*2, goal_tol*2, color=goal_color))

custom_lines = [Line2D([0], [0], color=hjb_color, lw=4),
                Line2D([0], [0], color=our_color, lw=4),
                Line2D([0], [0], color=goal_color, lw=4)]


ax.legend(custom_lines, ["HJB", "Our Method", "Target Set"])
os.makedirs("../figures", exist_ok=True)
fig.savefig("../figures/figure2.png")
