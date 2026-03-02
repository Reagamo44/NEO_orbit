# Extended Kalman Filter (EKF) implementation for orbit determination.

import numpy as np
import scipy as sp

from navigation.dynamics import rhs_2bod, MU_Earth
from navigation.propagate import propagate, numerical_jacobian
from navigation.measurement import range_measurement, range_jacobian


def ekf_step(state0, P, dt, *, mu, Q = None, jac_eps=1e-6):
    """
    Perform one step of the EKF: predict and update.
    """
    x = np.asarray(state0, dtype=float).reshape(-1) # Ensure state is a 1D array
    P = np.asarray(P, dtype=float) # Ensure covariance is a 2D array

    if x.size != 6:
        raise ValueError("State vector must have 6 elements (position and velocity).") # Ensure state vector has 6 elements
    if P.shape != (6, 6):
        raise ValueError("Covariance matrix P must be 6x6.") # Ensure P is 6x6
    
    if Q is None:
        Q = np.zeros((6, 6), dtype=float) # Default to zero process noise if not provided
    else: 
        Q = np.asarray(Q, dtype=float) # Ensure Q is a 2D array

        if Q.shape != (6, 6):
            raise ValueError("Process noise covariance Q must be 6x6.") 
        
    x_pred = propagate(x, dt, rhs_2bod, mu=mu) # Predict the next state using the dynamics model
    F = numerical_jacobian(lambda s: propagate(s, dt, rhs_2bod, mu=mu), x, eps=jac_eps) # Compute the Jacobian of the dynamics for covariance propagation

    P_pred = F @ P @ F.T + Q # Propagate the covariance using the Jacobian and add process noise

    P_pred = (P_pred + P_pred.T) / 2 # Ensure the covariance matrix remains symmetric
    return x_pred, P_pred # Return the predicted state and covariance



