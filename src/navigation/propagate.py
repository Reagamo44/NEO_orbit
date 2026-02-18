import numpy as np
import scipy as sp
from navigation.dynamics import rhs_2bod, MU_Earth

def propagate(state0: np.ndarray, 
              dt: float, mu: float = MU_Earth, 
              method: str = "DOP853" # default to a high-order Runge-Kutta method
              ) -> np.ndarray:
    """
    Propagate the state of a two-body system over a time step dt.
    """

    state0 = np.asarray(state0, dtype=float).reshape(6)  # ensure state is a 6-element vector

    sol = sp.integrate.solve_ivp(
                       fun = lambda t,
                       state: rhs_2bod(t, state, mu = mu),
                       t_span = (0.0, float(dt)),
                       y0 = state0,
                       method = method)
    
    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")
    return sol.y[:, -1].copy() # return the final state after propagation

def numerical_jacobian(state: np.ndarray, 
                       dt: float, 
                       mu: float = MU_Earth, 
                       epsilon: float = 1e-6 # small perturbation for finite difference
                    ) -> np.ndarray:
    """
    Compute the numerical Jacobian of the propogation function using finite differences.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    jacobian = np.zeros((6, 6))

    for i in range(6):
        perturb = np.zeros(6)
        scale = max(abs(state[i]), 1.0)  # scale perturbation based on the magnitude of the state variable
        perturb[i] = epsilon * scale # perturbation in the i-th direction
        delta = perturb[i]
        state_plus = state + perturb
        state_minus = state - perturb

        f_plus = propagate(state_plus, dt, mu)
        f_minus = propagate(state_minus, dt, mu)

        jacobian[:, i] = (f_plus - f_minus) / (2 * delta) # central difference approximation
        eigvals = np.linalg.eigvals(jacobian) # compute eigenvalues of the Jacobian
        if np.any(np.abs(eigvals) > 1e3):
            print(f"Large eigenvalues: {eigvals}")


    return jacobian