import numpy as np
from scipy.linalg import expm


def evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function, t_0=0,
                                 stop_criterion='steps', num_steps=None, t_max=None):
    """
    Advance a charged particle in time under the influence of E,B fields.
    """
    if stop_criterion == 'time':
        num_steps = t_max / dt
    x_list = [x_0]
    v_list = [v_0]
    t_list = [t_0]
    t = t_0
    for i in range(num_steps):
        x_new, v_new = particle_integration_step(x_list[-1], v_list[-1], t_list[-1],
                                                 dt, E_function, B_function)
        x_list += [x_new]
        v_list += [v_new]
        t += dt
        t_list += [t]

    t = np.array(t_list)
    x_list = np.array(x_list)
    v_list = np.array(v_list)
    x = x_list[:, 0]
    y = x_list[:, 1]
    z = x_list[:, 2]
    vx = v_list[:, 0]
    vy = v_list[:, 1]
    vz = v_list[:, 2]
    return t, x, y, z, vx, vy, vz


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


def get_radius(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)
