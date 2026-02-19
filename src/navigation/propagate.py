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
                       fun = lambda t, # time variable (not used in two-body problem but required by ODE solvers)
                       state: rhs_2bod(t, state, mu = mu),
                       t_span = (0.0, float(dt)), # [s] time span for integration
                       t_eval = [float(dt)], # evaluate the solution at the final time only
                       y0 = state0, # vector of initial conditions
                       method = method, # use the specified integration method
                       rtol = 1e-10, # relative tolerance for the solver
                       atol = 1e-12 # absolute tolerance for the solver
                   )
  
    
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
    J = np.zeros((6, 6)) # initialize Jacobian matrix

    for i in range(6):
        perturb = np.zeros(6) # perturbation in state units: [m] for r, [m/s] for v
        scale = max(abs(state[i]), 1.0)  # scale perturbation based on the magnitude of the state variable
        perturb[i] = epsilon * scale # perturbation in the i-th direction
        delta = perturb[i] # actual perturbation magnitude
        state_plus = state + perturb # state with positive perturbation
        state_minus = state - perturb # state with negative perturbation

        f_plus = propagate(state_plus, dt, mu) # propagated 6-state with positive perturbation
        f_minus = propagate(state_minus, dt, mu) # propagated 6-state with negative perturbation

        J[:, i] = (f_plus - f_minus) / (2 * delta) # discrete-time STM column (units vary by block)
    
    # Check for large eigenvalues which may indicate numerical instability
    eigvals = np.linalg.eigvals(J) # compute eigenvalues of the J
    if np.any(np.abs(eigvals) > 1e3):
        print(f"Large eigenvalues: {eigvals}")

    return J