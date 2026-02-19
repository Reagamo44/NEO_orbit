import numpy as np
import matplotlib.pyplot as plt
from navigation.dynamics import MU_Earth, R_Earth
from navigation.propagate import propagate, numerical_jacobian


def initial_elliptical_state(alt_p: float = 500e3, alt_a: float = 2000e3, R: float = R_Earth, mu: float = MU_Earth    ) -> np.ndarray:

    """
    Generate an initial state vector for an elliptical orbit.
    """
    rp = R + alt_p  # [m] periapsis radius
    ra = R + alt_a  # [m] apoapsis radius

    a = (rp + ra) / 2  # [m] semi-major axis
    e = (ra - rp) / (ra + rp)  # eccentricity

    vp = np.sqrt(mu * (2/rp - 1/a))  # [m/s] velocity at periapsis

    # Initial state: position (x, y, z) and velocity (vx, vy, vz)
    return np.array([rp, 0, 0, 0, vp, 0]), mu, ra, rp # starting at (r_norm, 0, 0) with velocity (0, v_circular, 0)

def test_elliptical_orbit():

    # Parameters for a circular orbit
    state0, mu, ra, rp = initial_elliptical_state()
    r0 = state0[0:3]  # [m] initial orbital radius
    v0 = state0[3:6]  # [m/s] initial orbital velocity
    r0_norm = np.linalg.norm(r0) # [m] magnitude of the initial orbital radius
    v0_norm = np.linalg.norm(v0) # [m/s] magnitude of the initial orbital velocity

    dt = 60  # time step in seconds
    T = 2*np.pi*r0_norm / v0_norm # orbital period for circular orbit
    num_steps = round(4*T/dt) # simulate for 4 orbital periods
    
    h0 = np.cross(r0, v0) # [m^2/s] specific angular momentum vector
    h0_norm = np.linalg.norm(h0) # [m^2/s] magnitude of the specific angular momentum

    eps0 = 0.5 * v0_norm**2 - mu / r0_norm # [J/kg] initial specific orbital energy

    state = state0.copy() # initialize state for propagation

    energy = []
    angular_momentum = []

    max_h_dev = 0.0 #[m^2/s] maximum deviation of angular momentum from initial value
    max_eps_dev = 0.0 # [J/kg] maximum deviation of energy from initial value
    r_min = 1e99 # [m] initialize minimum orbital radius to a very large value
    r_max = 0.0 # [m] initialize maximum orbital radius to zero

    for _ in range(num_steps):
        state = propagate(state, dt) # propagate the state by one time step
        r = np.linalg.norm(state[0:3]) # [m] compute the current orbital radius
        v = np.linalg.norm(state[3:6]) # compute the current orbital velocity

        eps = 0.5 * v**2 - mu / r #[J/kg] compute the specific orbital energy
        h = np.cross(state[0:3], state[3:6]) # [m^2/s] specific angular momentum vector
        h_norm = np.linalg.norm(h) # [m^2/s] magnitude of the specific angular momentum


        energy.append(eps) #track specific orbital energy
        angular_momentum.append(h_norm) # track the magnitude of the specific angular momentum

        max_h_dev = max(max_h_dev, abs(h_norm - h0_norm)) # maximum deviation of angular momentum from initial value
        max_eps_dev = max(max_eps_dev, abs(eps - eps0)) # maximum deviation of energy from initial value

        r_min = min(r_min, r) # track minimum orbital radius
        r_max = max(r_max, r) # track maximum orbital radius    

    assert max_h_dev < 1e-2, f"Angular momentum deviated from initial value: {angular_momentum[-1]} m^2/s" # allow for small numerical deviations in angular momentum
    assert max_eps_dev < 1e-3, f"Energy deviated from initial value: {energy[-1]} J/kg" # allow for small numerical deviations in energy

    assert abs(r_min - rp) < 1e3, f"Perigee radius deviated from initial value: {r_min} m" # allow for small numerical deviations in perigee radius
    assert abs(r_max - ra) < 1e3, f"Apogee radius deviated from initial value: {r_max} m" # allow for small numerical deviations in apogee radius

    # Plot the results
    plt.figure()
    plt.plot(angular_momentum)
    plt.title("Angular Momentum over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Angular Momentum [m^2/s]")

    plt.figure()
    plt.plot(energy)
    plt.title("Specific Energy over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Specific Energy [J/kg]")
    
if __name__ == "__main__":
    test_elliptical_orbit()
    print("Test passed: Elliptical orbit is maintained.")