## main path equations ##
import numpy as np

MU_Earth = 3.986004418e14  # [m^3/s^2] gravitational parameter of Earth
R_Earth = 6.371e6  # [m] mean radius of Earth

def a_2bod(r: np.ndarray, mu: float = MU_Earth) -> np.ndarray:
    """
    Calculate the acceleration due to gravity for the two-body problem.
    """

    r_norm = np.linalg.norm(r) # [m] magnitude of the position vector
    if r_norm == 0:
        raise ValueError("Position vector cannot be zero.")
    return -mu * r / r_norm**3


def rhs_2bod(t: float, state: np.ndarray, mu: float = MU_Earth) -> np.ndarray:
    """
    Calculate the right-hand side of the two-body problem.
    """
    del t # time is not used in the two-body problem, but is included compatibility with ODE solvers
    r_vec = state[0:3]  # [m] position vector
    v_vec = state[3:6]  # [m/s] velocity vector
    a_vec = a_2bod(r_vec, mu)  # [m/s^2] acceleration vector
    return np.hstack((v_vec, a_vec))  # return combined velocity and acceleration



    
