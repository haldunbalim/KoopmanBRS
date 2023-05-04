import random

class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
    
class State(object):
    limits = None

    def __init__(self):
        pass

    @classproperty
    def ndim(cls):
        return cls.limits.shape[1]

    def __array__(self):
        raise Exception("Base Class Error")

    def get_f(self):
        raise Exception("Base Class Error")

    def get_g(self):
        raise Exception("Base Class Error")

    @classmethod
    def generate_random(cls):
        limits = cls.limits
        return cls(*(random.uniform(*limits[:, i]) for i in range(limits.shape[1])))


class Action:
    limits = None

    def __init__(self):
        pass

    @classproperty
    def ndim(cls):
        return cls.limits.shape[1]

    def __array__(self):
        raise Exception("Base Class Error")

    @staticmethod
    def generate_zero():
        raise Exception("Base Class Error")

    @classmethod
    def generate_random(cls):
        limits = cls.limits
        return cls(*(random.uniform(*limits[:, i]) for i in range(limits.shape[1])))


class DynamicsModel:
    state_class = None
    action_class = None
    dt = 5e-2

    def get_viz_fig(self):
        raise Exception("Base Class Error")

    def visualize_step():
        raise Exception("Base Class Error")

    def __repr__(self):
        return self.__class__.__name__

    def generate_random_state(self):
        return self.state_class.generate_random()

    def generate_random_action(self):
        return self.action_class.generate_random()

    def generate_zero_action(self):
        return self.action_class.generate_zero()

    def generate_init_state(self) -> State:
        return self.generate_random_state()

    def generate_goal_state(self) -> State:
        return self.generate_random_state()

    def goal_reached(self, state, goal_state):
        raise Exception("Base Class Error")

    def inverse_sample(self, next_state):
        raise Exception("Base Class Error")

    @staticmethod
    def get_true_dynamics_casadi(x, u):
        raise Exception("Base Class Error")

    @staticmethod
    def get_state_diff_casadi(this, other):
        raise Exception("Base Class Error")

    @classproperty
    def state_limits(self):
        return self.state_class.limits

    @classproperty
    def state_weights(self):
        return self.state_class.weights

    @classproperty
    def state_terminal_weights(self):
        return self.state_class.terminal_weights

    @classproperty
    def action_limits(self):
        return self.action_class.limits

    @classproperty
    def action_weights(self):
        return self.action_class.weights
    


