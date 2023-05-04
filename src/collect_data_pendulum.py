import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from itertools import product
from simulation import Simulator, PendulumModel, PendulumState
from utils import set_seed


num_val_tuples = 50000
dyn_model_st = "pendulum"
grid_ds = [0.04, 0.04, 0.08]
val_b = 0.05
integrate = False
set_seed(0)

states, actions, next_states = [], [], []
dyn_model = PendulumModel()
print(f"Collecting data for {dyn_model_st.title()}")
sa_limits = np.concatenate([dyn_model.state_limits, dyn_model.action_limits], axis=1)
grid_pts = np.array(
                list(product(*[np.arange(_l, _u+gds, gds) for _l, _u, gds in zip(*sa_limits, grid_ds)])))
for th, thdot, u in tqdm(grid_pts):
    state = dyn_model.state_class(th, thdot)
    action = dyn_model.action_class(u)
    next_state = Simulator.successor_ivp(dyn_model, state, action, bound=False, integrate=integrate)
    states.append(np.array(state))
    actions.append(np.array(action))
    next_states.append(np.array(next_state))

train_states, train_actions, train_next_states = np.array(states), np.array(actions), np.array(next_states)

states, actions, next_states = [], [], []
for i in tqdm(range(num_val_tuples // 2)):
    state = dyn_model.generate_random_state()
    action = dyn_model.generate_random_action()
    next_state = Simulator.successor_ivp(dyn_model, state, action, bound=False, integrate=integrate)
    states.append(np.array(state))
    actions.append(np.array(action))
    next_states.append(np.array(next_state))

    state = PendulumState(state.psi+np.random.rand()*val_b, state.psidot+np.random.rand()*val_b)
    action = dyn_model.generate_random_action()
    next_state = Simulator.successor_ivp(dyn_model, state, action, bound=False, integrate=integrate)
    states.append(np.array(state))
    actions.append(np.array(action))
    next_states.append(np.array(next_state))

val_states, val_actions, val_next_states = np.array(states), np.array(actions), np.array(next_states)


folder = os.path.join("..", "data", dyn_model_st)
os.makedirs(folder, exist_ok=True)
with open(os.path.join(folder, "train_single.pkl"), "wb") as f:
    pickle.dump({"states": train_states, "actions": train_actions, "next_states":train_next_states}, f)
with open(os.path.join(folder, "val_single.pkl"), "wb") as f:
    pickle.dump({"states": val_states, "actions": val_actions, "next_states":val_next_states}, f)