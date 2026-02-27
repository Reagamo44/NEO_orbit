
## For now, it is assumed the Earth is not rotating and the station is fixed in ECI coordinates. 
# In the future, Earth rotation and transform to ECEF coordinates can be added for more realistic measurements.

import numpy as np
from navigation.dynamics import R_Earth

def station_position():
    """
    Return the position of the ground station in ECI coordinates.
    For simplicity, the station is located at the equator at longitude 0.
    """
    return np.array([R_Earth, 0, 0]) # [m] position of the station in ECI coordinates

def range_measurement(state: np.ndarray, r_station: np.ndarray) -> float:
    """
    Compute the range measurement from the station to the object.
    """
    state = np.asarray(state, dtype=float).reshape(-1) # ensure state is a 1D array of floats
    if state.size != 6:
        raise ValueError("state must be length 6: [rx, ry, rz, vx, vy, vz]")
    
    r_object = state[0:3] # [m] position of the object in ECI coordinates
    
    r_station = np.asarray(r_station, dtype=float).reshape(3) # ensure station position is a 1D array of length 3
    
    dr = r_object - r_station # [m] range vector from station to object
    return float(np.linalg.norm(dr)) # [m] range measurement (magnitude of the range vector)


def range_jacobian(state: np.ndarray, r_station: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian of the range measurement with respect to the state.
    """
    
    state = np.asarray(state, dtype=float).reshape(-1) # ensure state is a 1D array of floats
    if state.size != 6:
        raise ValueError("state must be length 6: [rx, ry, rz, vx, vy, vz]")
    
    r_object = state[0:3] # [m] position of the object in ECI coordinates
    r_station = np.asarray(r_station, dtype=float).reshape(3) # ensure station position is a 1D array of length 3
    delta_r = r_object - r_station # [m] range vector from station to object

    rho = np.linalg.norm(delta_r) # [m] magnitude of the range vector

    if rho < 1e-12:
        raise ValueError("Range cannot be zero for Jacobian computation.")

    H = np.zeros((1, 6)) # initialize Jacobian matrix
    H[0, 0:3] = delta_r / rho # partial derivatives with respect to position

    return H
