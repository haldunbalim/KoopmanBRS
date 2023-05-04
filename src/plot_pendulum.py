from utils.geometry import minkowski_diff_poly_box
import cvxpy as cp
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.io import loadmat
import pickle
from simulation import PendulumModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.geometry import minkowski_diff_poly_box
import numpy as np

dyn_model = PendulumModel()
state_limits = dyn_model.state_limits
action_limits = dyn_model.action_limits
goal_psi_tol = 0.1
goal_psidot_tol = 0.5

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
        
model_sin = LiftingModel(0)
model_id = LiftingModel(1)

def get_levels(xv, yv, all_sets_enumerated, encode_fn):
    zv = []
    for th, thdot in tqdm(zip(xv.reshape(-1), yv.reshape(-1)), total=xv.size):
        point = np.array([th, thdot])
        encoded = encode_fn(point)
        zv.append(0)
        for i, s in all_sets_enumerated:
            if np.all(s.A@encoded <= s.b):
                zv[-1] = i+1
                break
    zv = np.array(zv)
    zv = zv.reshape(*xv.shape)
    return zv


data = loadmat("../data/hjb_pendulum.mat")["data"]
ths = np.linspace(*state_limits[:, 0], data.shape[0])
thdots = np.linspace(*state_limits[:, 1], data.shape[1])

xv_hjb, yv_hjb = np.meshgrid(ths, thdots, indexing='ij')

ths = np.linspace(*state_limits[:, 0], data.shape[0])
thdots = np.linspace(*state_limits[:, 1], data.shape[1])

xv_our, yv_our = np.meshgrid(ths, thdots, indexing='ij')

with open("../outputs/pendulum_brs_lifting=0.pkl", "rb") as f:
    algo_sin = pickle.load(f)["algo"]
    print("Generating BRS levels for sin lifting...")
    zv_sin = get_levels(xv_our, yv_our, algo_sin.all_sets_enumerated, model_sin.encode)

with open("../outputs/pendulum_brs_lifting=1.pkl", "rb") as f:
    algo_id= pickle.load(f)["algo"]
    print("Generating BRS levels for identity lifting...")
    zv_id = get_levels(xv_our, yv_our, algo_id.all_sets_enumerated, model_id.encode)

# -------- Control ---------

init_state = np.array([np.radians(180), 0])
init_state_encoded = model_sin.encode(init_state)
target_node = None
for i, tt in enumerate(algo_sin.targets_traj):
    for j, s in enumerate(tt):
        if np.all(s.A@init_state_encoded <= s.b):
            print("Minimum number of steps required", i)
            target_node = algo_sin.targets_traj_nodes[i-1][j]
            break
    if target_node:
        break
    if i == len(algo_sin.targets_traj)-1:
        raise Exception("Point is Outside")
    
def control(encoded, target_node):

    target_set = target_node.set_rep
    target_set_shrinked = minkowski_diff_poly_box(target_set,
                                                  target_node.error_bound_r,
                                                  target_node.error_bound_ct)

    A, B = target_node.A, target_node.B
    u = cp.Variable(1)
    nxt = A@encoded + B@u

    u_lo, u_hi = action_limits
    constraints = [target_set_shrinked.A@nxt <=
                   target_set_shrinked.b, u_hi >= u, u_lo <= u]
    prob = cp.Problem(cp.Minimize(cp.norm(u)), constraints)
    prob.solve()
    return u.value, nxt.value


curr = init_state_encoded
curr_target = target_node

states = [curr[:2]]
actions = []
while curr_target is not None:
    u, nxt_pred = control(curr, curr_target)
    curr = model_sin.encode(nxt_pred[:2])
    states.append(nxt_pred[:2])
    actions.append(u)
    curr_target = curr_target.parent

states = np.array(states)
actions = np.array(actions)

# -------- Visualization ---------
fig = plt.figure()
ax = fig.add_subplot()
hjb_color = "green"
goal_color = "orange"
sin_color = "lightblue"
id_color = "pink"

ax.contour(xv_hjb, yv_hjb, data[:, :, -1] < 0, colors=[hjb_color])

ax.plot(states[:, 0], states[:, 1], color="darkblue")
custom_lines = [Line2D([0], [0], color=hjb_color, lw=4),
                Line2D([0], [0], color=sin_color, lw=4),
                Line2D([0], [0], color=id_color, lw=4),
                Line2D([0], [0], color=goal_color, lw=4),
                Line2D([0], [0], color="darkblue", lw=4)]

cond = zv_sin > 0
ax.scatter(xv_our[cond].reshape(-1), yv_our[cond].reshape(-1), color=sin_color, alpha=1, s=2)

cond = zv_id > 0
ax.scatter(xv_our[cond].reshape(-1), yv_our[cond].reshape(-1), color=id_color, alpha=0.5, s=2)


ax.add_patch(Rectangle((0, -goal_psidot_tol), goal_psi_tol*2, goal_psidot_tol*2, color=goal_color))
ax.legend(custom_lines, ["HJB", r"Ours $\psi(\mathbf{x})=(\mathbf{x}, \sin(\theta))$",
                         r"Ours $\psi(\mathbf{x})=\mathbf{x}$",
                         "Target Set", "Trajectory"], loc="upper left")

fs = 16
ax.set_xlabel(r"$\theta$", fontsize=fs)
ax.set_ylabel(r"$\dot\theta$", fontsize=fs)

fig.savefig("../figures/figure3.png")
