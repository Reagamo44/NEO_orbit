import numpy as np
import matplotlib.pyplot as plt

from navigation.propagate import propagate
from navigation.measurement import station_position, range_measurement, range_jacobian
from circular_orbit import initial_circular_state


def fd_range_jacobian(state, r_station, eps=1.0):
    '''Compute the Jacobian of the range measurement with respect to the state using finite differences.'''

    state = np.asarray(state, dtype=float).reshape(-1) # ensure state is a 1D array of floats
    if state.size != 6:
        raise ValueError("state must be length 6")
    
    H = np.zeros((1, 6)) # initialize Jacobian matrix

    for i in range(6):
        d = np.zeros(6) # [m] perturbation vector
        d[i] = eps # perturb the i-th state variable by eps

        rp = range_measurement(state + d, r_station) # [m] range measurement with positive perturbation
        rm = range_measurement(state - d, r_station) # [m] range measurement with negative perturbation

        H[0, i] = (rp - rm) / (2 * eps) # finite difference approximation of the Jacobian column
    return H


def test_range_measurement(plot = True):
    '''Test the range measurement function and its Jacobian.'''

    sc_state0, mu = initial_circular_state() # [m] initial state of the spacecraft in ECI coordinates
    r0_sc_norm = np.linalg.norm(sc_state0[0:3]) # [m] magnitude of the initial orbital radius
    v0_sc_norm = np.linalg.norm(sc_state0[3:6]) # [m/s] magnitude of the initial orbital velocity

    r_st = station_position() # [m] position of the station in ECI coordinates

    dt = 60.0  # time step in seconds
    T = 2*np.pi*r0_sc_norm / v0_sc_norm # orbital period for circular orbit
    num_steps = int(np.round(T/dt)) # simulate for 1 orbital period
    
    sigma = 5000.0  # [m] standard deviation of measurement noise
    rho_true = [] # [m] true range measurements without noise
    rho_meas = [] # [m] measured range with noise
    t = [] # [s] time vector for plotting

    sc_state = sc_state0.copy() # initialize state for propagation

    for i in range(num_steps + 1):
        
        rho = range_measurement(sc_state, r_st) # [m] range measurement from station to spacecraft
        rho_true.append(rho) # [m] store the true range measurement without noise
        rho_meas.append(rho + np.random.normal(0, sigma)) # add measurement noise
        t.append(i*dt) # [s] time for plotting

        sc_state = propagate(sc_state, dt, mu=mu) # propagate the state by one time step

    rho_true = np.array(rho_true)
    rho_meas = np.array(rho_meas)
    t = np.array(t)

    assert np.all(rho_true > 0.0), "True range measurements should be positive."
    
    noise = rho_meas - rho_true
    assert abs(noise.mean()) < 0.2*sigma, f"Measurement noise mean is too large: {noise.mean():.2f} m"
    assert abs(noise.std(ddof=1) - sigma) < 0.2*sigma, f"Measurement noise standard deviation is too far from expected: {noise.std(ddof=1):.2f} m vs {sigma:.2f} m"

    # Jacobian check at a representative state (mid-arc)
    mid_state = sc_state0.copy()
    for _ in range(10):
        mid_state = propagate(mid_state, dt, mu=mu)

    H_a = range_jacobian(mid_state, r_st)
    H_n = fd_range_jacobian(mid_state, r_st, eps=1.0)  # 1 m perturbation
    print(f"Analytic Jacobian:\n{H_a}\nNumeric Jacobian:\n{H_n}")
    print(f"Difference:\n{np.max(np.abs(H_n - H_a))}")
    print(f"Mean noise: {np.mean(noise):.2f} m, Std noise: {np.std(noise, ddof=1):.2f} m")

    assert np.allclose(H_a, H_n, rtol=1e-7, atol=1e-9), f"\nAnalytic:\n{H_a}\nNumeric:\n{H_n}"

    if plot:
        plt.figure()
        plt.plot(t, rho_true, label="True Range")
        plt.plot(t, rho_meas, label="Measured Range", linestyle='dashed')
        plt.title("Range Measurement over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Range [m]")
        plt.legend()

        plt.figure()
        plt.plot(t, noise)
        plt.title("Measurement Residual (Noise) over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Range Error [m]")
        plt.axhline(0, color='k', linewidth=1)
        plt.show()

if __name__ == "__main__":
    np.random.seed(0) # set random seed for reproducibility
    test_range_measurement(plot=True)
