import numpy as np
from scipy.integrate import solve_ivp

class Simulator:
    @staticmethod
    def successor_ivp(dyn_model, state, action, integrate=False, bound=True):
        def dynamics(t, y):
            state = dyn_model.state_class(*y[:dyn_model.nx])
            action = dyn_model.action_class(*y[dyn_model.nx:])
            dx = _dynamics(state, action)
            du = np.zeros(dyn_model.nu)
            return np.concatenate([dx, du])
        
        def _dynamics(state, action):
            f = state.get_f()
            g = state.get_g()
            dx = f + g.dot(np.array(action))
            return dx

        if integrate:
            y0 = np.concatenate([state, action])
            result = solve_ivp(fun=dynamics, t_span=(
                0.0, float(dyn_model.dt)), y0=y0, method='RK45')
            if not result.success:
                raise RuntimeError("Failed to integrate ivp!")
            xnext = result.y[:dyn_model.nx, -1]
        else:
            dx = _dynamics(state, action)
            # first order taylor approximation
            xnext = np.array(state) + dx * dyn_model.dt
        return dyn_model.state_class(*xnext) if bound else xnext


