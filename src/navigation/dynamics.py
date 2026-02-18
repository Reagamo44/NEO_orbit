## main path equations ##
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def grav_acceleration(m, r):
    """Calculate the gravitational acceleration on a mass m at distance r from a mass M."""
    G = 6.67430e-11  # gravitational constant
    return G * m / r**2


def rhs_2bod(t, m, state):
    """Calculate the right-hand side of the two-body problem."""
    r_vec = state[0:3]  # position vector
    v_vec = state[3:6]  # velocity vector
    r = np.linalg.norm(r_vec)  # distance between the two bodies
    v = np.linalg.norm(v_vec)  # speed of the body
    a_vec = -grav_acceleration(m, r) * r_vec / r  # acceleration vector
    
    return v_vec, a_vec

def move_2bod(t, m, state):
    """Calculate the new state of the system after a time step dt."""
    v_vec, a_vec = rhs_2bod(t, m, state)
    new_state = np.zeros_like(state)
    new_state[0:3] = state[0:3] + v_vec * dt  # update position
    new_state[3:6] = state[3:6] + a_vec * dt  # update velocity
    
    return new_state


    
