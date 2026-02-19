import numpy as np
import matplotlib.pyplot as plt
from navigation.dynamics import MU_Earth, R_Earth
from navigation.propagate import propagate, numerical_jacobian


def initial_circular_state(altitude: float = 500e3, R: float = R_Earth, mu: float = MU_Earth    ) -> np.ndarray:

    """
    Generate an initial state vector for a circular orbit at a given altitude.
    """
    r_norm = R + altitude  # [m] orbital radius
    v_circular = np.sqrt(mu / r_norm)  # [m/s] circular orbital velocity

    # Initial state: position (x, y, z) and velocity (vx, vy, vz)
    return np.array([r_norm, 0, 0, 0, v_circular, 0]), mu  # starting at (r_norm, 0, 0) with velocity (0, v_circular, 0)

def test_circular_orbit():

    # Parameters for a circular orbit
    state0, mu = initial_circular_state()
    r0 = state0[0:3]  # [m] initial orbital radius
    v0 = state0[3:6]  # [m/s] initial orbital velocity
    r0_norm = np.linalg.norm(r0) # [m] magnitude of the initial orbital radius
    v0_norm = np.linalg.norm(v0) # [m/s] magnitude of the initial orbital velocity

    dt = 60  # time step in seconds
    T = 2*np.pi*r0_norm / v0_norm # orbital period for circular orbit
    num_steps = round(4*T/dt) # simulate for 4 orbital periods

    state = state0.copy() # initialize state for propagation
    radius = []
    energy = []

    for _ in range(num_steps):
        state = propagate(state, dt) # propagate the state by one time step
        r = np.linalg.norm(state[0:3]) # compute the current orbital radius
        v = np.linalg.norm(state[3:6]) # compute the current orbital velocity

        radius.append(r)
        energy.append(0.5 * v**2 - mu / r) # compute the specific orbital energy

        assert np.max(np.abs(radius - r0_norm)) < 10, f"Radius deviated from initial value: {radius[-1]} m"
        assert np.max(np.abs(energy - (0.5 * v0_norm**2 - mu / r0_norm))) < 1e-3, f"Energy deviated from initial value: {energy[-1]} J/kg"

    # Plot the results
    plt.figure()
    plt.plot(radius)
    plt.title("Orbital Radius over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Orbital Radius [m]")

    plt.figure()
    plt.plot(energy)
    plt.title("Specific Energy over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Specific Energy [J/kg]")
    plt.show()
    
if __name__ == "__main__":
    test_circular_orbit()
    print("Test passed: Circular orbit is maintained.")