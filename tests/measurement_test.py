import numpy as np
import matplotlib.pyplot as plt

from navigation.dynamics import MU_Earth, R_Earth
from navigation.propagate import propagate, numerical_jacobian
from navigation.measurement import station_position, range_measurement, range_jacobian
from circular_orbit import initial_circular_state


def test_range_measurement(plot = True):
    '''Test the range measurement function and its Jacobian.'''

    sc_state0, mu = initial_circular_state() # [m] initial state of the spacecraft in ECI coordinates
    r0_sc = sc_state0[0:3]  # [m] initial orbital radius
    v0_sc = sc_state0[3:6]  # [m/s] initial orbital velocity
    r0_sc_norm = np.linalg.norm(r0_sc) # [m] magnitude of the initial orbital radius
    v0_sc_norm = np.linalg.norm(v0_sc) # [m/s] magnitude of the initial orbital velocity

    r_st = station_position() # [m] position of the station in ECI coordinates

    dt = 60  # time step in seconds
    T = 2*np.pi*r0_sc_norm / v0_sc_norm # orbital period for circular orbit
    num_steps = round(4*T/dt) # simulate for 4 orbital periods
    
    sigma = 100.0  # meters
    rho_true = []
    rho_meas = []
    t = []

    sc_state = sc_state0.copy() # initialize state for propagation

    for i in range(num_steps + 1):
        
        rho = range_measurement(sc_state, r_st) # [m] range measurement from station to spacecraft
        rho_true.append(rho)
        rho_meas.append(rho + np.random.normal(0, sigma)) # add measurement noise
        t.append(i*dt)

        sc_state = propagate(sc_state, i*dt, mu=mu) # propagate the state by i time steps

    rho_true = np.array(rho_true)
    rho_meas = np.array(rho_meas)
    t = np.array(t)

    assert np.all(rho_true > 0.0), "True range measurements should be positive."
    
    noise = rho_meas - rho_true
    assert np.max(np.abs(noise)) < 6*sigma, "Noise should be within 6 sigma of true range."

    # Jacobian check at a representative state (mid-arc)
    mid_state = sc_state0.copy()
    for _ in range(10):
        mid_state = propagate(mid_state, dt, mu=mu)

    H_a = range_jacobian(mid_state, r_st)
    H_n = range_jacobian(mid_state, r_st, eps=1.0)  # 1 m perturbation
    assert np.allclose(H_a, H_n, rtol=1e-7, atol=1e-9), f"\nAnalytic:\n{H_a}\nNumeric:\n{H_n}"

    if plot:
        plt.figure()
        plt.plot(t, rho_true, label="True Range")
        plt.plot(t, rho_meas, label="Measured Range", linestyle='dashed')
        plt.title("Range Measurement over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Range [m]")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    np.random.seed(0) # set random seed for reproducibility
    test_range_measurement(plot=True)
