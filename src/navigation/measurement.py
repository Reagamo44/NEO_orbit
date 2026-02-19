
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
    r_object = state[0:3] # [m] position of the object in ECI coordinates

    return (r_object - r_station), r_object # [m] range measurement


def range_jacobian(state: np.ndarray, r_station: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian of the range measurement with respect to the state.
    """
    delta_r, r_object = range_measurement(state, r_station) # [m] range vector from station to object and object poisition
    rho = np.linalg.norm(delta_r) # [m] magnitude of the range vector

    if rho == 0:
        raise ValueError("Range cannot be zero for Jacobian computation.")

    H = np.zeros((1, 6)) # initialize Jacobian matrix
    H[0, 0:3] = delta_r / rho # partial derivatives with respect to position

    return H
