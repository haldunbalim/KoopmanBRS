import numpy as np
from simulation import State, Action, DynamicsModel

class DuffingConfig:
    a = 2
    b = -2
    d = -.5

    dt = 0.025

    xmin = -0.5
    xmax = 0.5
    ymin = -1.5
    ymax = 1.5

    #DuffingState(0.5, 0)
    init_xmin = .40
    init_xmax = .60
    init_ymin = -.10
    init_ymax = .10

    max_action = 5
    min_action = -5
    goal_tol = 0.1

    # viz
    agent_viz_radius = min(xmax-xmin, ymax-ymin) / 50


class DuffingState(State):
    limits = np.array([[DuffingConfig.xmin, DuffingConfig.ymin],
                       [DuffingConfig.xmax, DuffingConfig.ymax]], dtype="f")

    def __init__(self, x, y):
        super().__init__()
        self.x = np.clip(x, *self.limits[:, 0])
        self.y = np.clip(y, *self.limits[:, 1])

    def __array__(self):
        return np.array([self.x, self.y]).astype("f")

    def __repr__(self):
        return "x: {:.3f}, y:{:.3f}".format(self.x, self.y)

    def normalize(self):
        return self

    def get_f(self):
        a, b, d = DuffingConfig.a, DuffingConfig.b, DuffingConfig.d
        x, y = self.x, self.y
        f = np.array([y,
                      a * x + b * x**3 + d * y], dtype="f")
        return f

    def get_g(self):
        g = np.zeros((2, 1), dtype="f")
        g[1] = 1
        return g


class DuffingAction(Action):
    # w = v/r
    limits = np.array([[DuffingConfig.min_action],
                       [DuffingConfig.max_action]], dtype="f")

    def __init__(self, u):
        super().__init__()
        self.u = np.clip(u, *self.limits[:, 0])

    def __repr__(self):
        return "{:.3f}".format(self.u)

    def __array__(self):
        return np.array([self.u]).astype(np.float32)

    @staticmethod
    def generate_zero():
        return DuffingAction(0)


class DuffingModel(DynamicsModel):

    state_class = DuffingState
    action_class = DuffingAction
    nx = state_class.ndim
    nu = action_class.ndim
    init_limits = np.array([[DuffingConfig.init_xmin, DuffingConfig.init_ymin],
                            [DuffingConfig.init_xmax, DuffingConfig.init_ymax]])
    dt = DuffingConfig.dt

    def __init__(self):
        super().__init__()

    def goal_reached(self, state, goal):
        return abs(state.x - goal.x) < DuffingConfig.goal_tol and \
               abs(state.y - goal.y) < DuffingConfig.goal_tol

    def generate_goal_state(self):
        return DuffingState(0, 0)

    def generate_init_state(self):
        x = np.random.uniform(DuffingConfig.init_xmin, DuffingConfig.init_xmax)
        y = np.random.uniform(DuffingConfig.init_ymin, DuffingConfig.init_ymax)
        return DuffingState(x, y)

