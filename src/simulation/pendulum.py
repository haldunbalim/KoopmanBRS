import numpy as np
from simulation import State, Action, DynamicsModel

class PendulumConfig:
    min_speed = -6
    max_speed = 4
    min_torque = -0.35
    max_torque = 0.35
    dt = 1e-1

    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3

    m = 0.1
    l = 1
    g = 10

    goal_psi_tol = 1e-1
    goal_psidot_tol = 5e-1
    # viz
    rod_len = 2
    rod_width = 1e-1


class PendulumState(State):
    limits = np.array([[0, PendulumConfig.min_speed],
                       [3*np.pi/2,  PendulumConfig.max_speed]], dtype="f")

    def __init__(self, psi, psidot):
        super().__init__()
        self.psi = np.clip(psi, *self.limits[:, 0])
        self.psidot = np.clip(psidot, *self.limits[:, 1])

    def __array__(self):
        return np.array([self.psi, self.psidot]).astype("f")

    def __repr__(self):
        return "psi: {:.3f}, psidot:{:.3f}".format(self.psi, self.psidot)

    def normalize(self):
        return PendulumState(self.psi % (2*np.pi), self.psidot)

    def get_f(self):
        g = PendulumConfig.g
        l = PendulumConfig.l
        f = np.array([self.psidot,
                      (3*g) / (2*l) * np.sin(self.psi)], dtype="f")
        return f

    def get_g(self):
        m = PendulumConfig.m
        l = PendulumConfig.l
        g = np.zeros((2, 1), dtype="f")
        g[1] = 3 / (m*l**2)
        return g


class PendulumAction(Action):
    # w = v/r
    limits = np.array([[PendulumConfig.min_torque],
                       [PendulumConfig.max_torque]], dtype="f")

    def __init__(self, torque):
        super().__init__()
        self.torque = np.clip(torque, *self.limits[:, 0])

    def __repr__(self):
        return "{:.3f}".format(self.torque)

    def __array__(self):
        return np.array([self.torque]).astype(np.float32)

    @staticmethod
    def generate_zero():
        return PendulumAction(0)


class PendulumModel(DynamicsModel):

    state_class = PendulumState
    action_class = PendulumAction
    nx = state_class.ndim
    nu = action_class.ndim
    dt = PendulumConfig.dt

    def __init__(self):
        super().__init__()

    def generate_goal_state(self):
        psi = 0
        psidot = 0
        return self.state_class(psi+PendulumConfig.goal_psi_tol, psidot-PendulumConfig.goal_psidot_tol)

    def generate_init_state(self):
        return PendulumState(np.pi*3/2, 0)
