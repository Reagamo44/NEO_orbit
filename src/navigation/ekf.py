# Extended Kalman Filter (EKF) implementation for orbit determination.

import numpy as np

from navigation.propagate import propagate, numerical_jacobian
from navigation.measurement import range_measurement, range_jacobian


def ekf_predict(state0, P, dt, *, mu, Q = None, jac_eps=1e-6):
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
        
    x_pred = propagate(x, dt, mu=mu) # Predict the next state using the dynamics model
    F = numerical_jacobian(x, dt, mu=mu, epsilon=jac_eps) # Compute the Jacobian of the dynamics for covariance propagation

    P_pred = F @ P @ F.T + Q # Propagate the covariance using the Jacobian and add process noise

    P_pred = (P_pred + P_pred.T) / 2 # Ensure the covariance matrix remains symmetric
    return x_pred, P_pred # Return the predicted state and covariance


def ekf_update(x_pred, P_pred, z_meas, r_station, *, R):
    '''Perform the EKF measurement update step using a range measurement.
    '''

    x = np.asarray(x_pred, dtype=float).reshape(-1) # Ensure predicted state is a 1D array
    P = np.asarray(P_pred, dtype=float) # Ensure predicted covariance is a 2

    if x.size != 6:
        raise ValueError("Predicted state vector must have 6 elements (position and velocity).") # Ensure predicted state has 6 elements
    if P.shape != (6, 6):
        raise ValueError("Predicted covariance matrix P must be 6x6.") # Ensure predicted covariance is 6x6
    
    z_hat = float(range_measurement(x, r_station)) # Compute the expected range measurement from the predicted state
    
    H = np.asarray(range_jacobian(x, r_station), dtype=float) # Compute the Jacobian of the
    if H.shape != (1, 6):
        raise ValueError("Measurement Jacobian H must be 1x6 for range measurement.") # Ensure H is 1x6
    
    y = float(z_meas - z_hat) # Compute the measurement residual (innovation)

    R = float(R) # Ensure measurement noise covariance is a scalar for range measurement

    S = float(H @ P @ H.T + R) # Compute the innovation covariance S (scalar for range measurement)
    if S <= 0.0 or not np.isfinite(S):
        raise ValueError("Innovation covariance S must be positive and finite.") # Ensure S is positive and finite  
    
    K = (P @ H.T) / S # Compute the Kalman gain

    x_upd = x + (K[:,0] * y) # Update the state estimate using the Kalman gain and measurement residual

    I = np.eye(6) # Identity matrix for covariance update
    KH = K @ H # Compute the Kalman gain times the measurement Jacobian
    P_upd = (I - KH) @ P @ (I - KH).T + (K * R) @ K.T # Update the covariance using the Joseph form for numerical stability
    P_upd = (P_upd + P_upd.T) / 2 # Ensure the updated covariance matrix remains symmetric

    return x_upd, P_upd, y, S # Return the updated state and covariance

def ekf_step_range(state0, P, dt, z_meas, r_station, *, mu, R, Q = None, jac_eps=1e-6):
    '''Perform one full EKF step: predict and update with a range measurement.'''
    
    x_pred, P_pred = ekf_predict(state0, P, dt, mu=mu, Q=Q, jac_eps=jac_eps) # Perform the prediction step
    x_upd, P_upd, y, S = ekf_update(x_pred, P_pred, z_meas, r_station, R=R) # Perform the update step with the range measurement
    
    return x_upd, P_upd, y, S # Return the updated state and covariance after the EKF step

