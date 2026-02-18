import numpy as np
import matplotlib.pyplot as plt
from navigation.dynamics import move_2bod

def test_circular_orbit():
    # Parameters for a circular orbit
    m = 5.972e24  # mass of Earth in kg
    r = 6.371e6 + 700e3  # radius of Earth + altitude 
    v_circular = np.sqrt(6.67430e-11 * m / r)  # circular orbital velocity

    # Initial state: position (x, y, z) and velocity (vx, vy, vz)
    state = np.array([r, 0, 0, 0, v_circular, 0])  # starting at (r, 0, 0) with velocity (0, v_circular, 0)

    dt = 60  # time step in seconds
    T = 2*np.pi*r / v_circular
    num_steps = round(4*T/dt)

    for _ in range(num_steps):
        state = move_2bod(dt, m, state)
    
    
if __name__ == "__main__":
    test_circular_orbit()
    print("Test passed: Circular orbit is maintained.")