import numpy as np
from scipy.linalg import expm


def get_thermal_velocity(T, m, kB_eV):
    '''
    Calculate the thermal velocity of particles according to 1/2*m*v_th^2 = 3/2*kB*T
    T in eV, m in kg
    '''
    return np.sqrt(3.0 * kB_eV * T / m)


def get_cyclotron_angular_frequency(q, B, m):
    '''
    Calculate the angular frequency of cyclotron / Larmor precession
    q in Coulomb, B in Tesla, m in kg
    '''
    return np.abs(q * B / m)


def evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function, t_0=0, q=1.0, m=1.0,
                                 stop_criterion='steps', num_steps=None, t_max=None, return_fields=False):
    """
    Advance a charged particle in time under the influence of E,B fields.
    """
    if stop_criterion == 'time':
        num_steps = t_max / dt
    t = t_0

    # define a dictionary that will collect all the particles history as it goes
    hist = {}
    hist['x'] = [x_0]
    hist['v'] = [v_0]
    hist['t'] = [t_0]
    if return_fields is True:
        hist['E'] = [E_function(x_0, t_0)]
        hist['B'] = [B_function(x_0, t_0)]

    for i in range(num_steps):
        x_new, v_new = particle_integration_step(hist['x'][-1], hist['v'][-1], hist['t'][-1],
                                                 dt, E_function, B_function, q=q, m=m)
        hist['x'] += [x_new]
        hist['v'] += [v_new]
        t += dt
        hist['t'] += [t]

        if return_fields is True:
            hist['E'] += [E_function(x_new, t)]
            hist['B'] += [B_function(x_new, t)]

    for key in hist:
        hist[key] = np.array(hist[key])
    return hist


def particle_integration_step(x_0, v_0, t, dt, E_function, B_function, q=1.0, m=1.0):
    """
    Algorithm based on "2015 - He et al - Volume-preserving algorithms for charged particle dynamics"
    https://www.sciencedirect.com/science/article/pii/S0021999114007141
    """
    x_half = x_0 + dt * v_0 / 2.0
    t_half = t + dt / 2.0
    E_half = E_function(x_half, t_half)
    # E_half = E_function(x_half, t)
    v_minus = v_0 + dt * q / m / 2.0 * E_half
    B_half = B_function(x_half, t_half)
    # B_half = B_function(x_half, t)
    B_norm = np.linalg.norm(B_half)
    b_x = B_half[0] / B_norm
    b_y = B_half[1] / B_norm
    b_z = B_half[2] / B_norm
    b_half_tensor = np.array([[0, -b_z, b_y], [b_z, 0, -b_x], [-b_y, b_x, 0]])
    # omega_half = - q * B_norm / m # definition with minus from paper gives wrong right hand rule
    omega_half = q * B_norm / m
    v_plus = np.dot(expm(dt * omega_half * b_half_tensor), v_minus)
    v_new = v_plus + dt * q / m / 2.0 * E_half
    x_new = x_half + dt / 2.0 * v_new
    return x_new, v_new
