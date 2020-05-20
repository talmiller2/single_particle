import numpy as np
import scipy.special
from scipy.linalg import expm

mu0 = 4 * np.pi * 1e-7  # permeability constant [Henry/m]


# np.seterr(all='raise')

def get_current_loop_magnetic_field_cartesian(x, y, z, x0=0, y0=0, z0=0, loop_radius=1.0, loop_current=1.0):
    """
    some code copied from https://github.com/tedyapo/loopfield
    based on equations from https://ntrs.nasa.gov/search.jsp?R=20010038494
    cartesian coordinates
    magnetic field of current loop (xy plane), centered in x0,y0,z0 [m], loop radius r [m],
    current I [A], return magnetic field vector B_x, B_y, B_z in [Tesla]
    """

    # print('begin B calc')

    # Galilean transformation
    X = x - x0
    Y = y - y0
    Z = z - z0

    a = loop_radius  # current loop radius
    C = mu0 * loop_current / np.pi

    rho2 = X ** 2 + Y ** 2
    rho = np.sqrt(rho2)
    r2 = rho2 + Z ** 2
    r = np.sqrt(r2)
    alpha2 = a ** 2 + r2 - 2.0 * a * rho
    beta2 = a ** 2 + r2 + 2.0 * a * rho
    beta = np.sqrt(beta2)
    k2 = 1 - alpha2 / beta2
    E_k2 = scipy.special.ellipe(k2)
    K_k2 = scipy.special.ellipkm1(1 - k2)  # more efficient than ellipk(k2), note the different argument
    # K_k2 = scipy.special.ellipk(k2)

    denom_xy = 2.0 * alpha2 * beta * rho2
    with np.errstate(invalid='ignore'):
        numer_xy_factor = C * ((a ** 2 + r ** 2) * E_k2 - alpha2 * K_k2)

    numer_x = X * Z * numer_xy_factor
    B_x = numer_x / denom_xy

    numer_y = Y * Z * numer_xy_factor
    B_y = numer_y / denom_xy

    denom_z = 2.0 * alpha2 * beta
    with np.errstate(invalid='ignore'):
        numer_z = C * ((a ** 2 - r ** 2) * E_k2 + alpha2 * K_k2)
    B_z = numer_z / denom_z

    return B_x, B_y, B_z


def get_current_loop_magnetic_field_cylindrical(rho, z, z0=0, loop_radius=1.0, loop_current=1.0):
    """
    cylindrical coordinates
    """

    # Galilean transformation
    Z = z - z0

    a = loop_radius  # current loop radius
    C = mu0 * loop_current / np.pi

    alpha2 = a ** 2 + rho ** 2 + Z ** 2 + 2.0 * a * rho
    beta2 = a ** 2 + rho ** 2 + Z ** 2 - 2.0 * a * rho
    beta = np.sqrt(beta2)
    inds_positive = np.where(beta > 0)
    k2 = beta * 0 + np.inf
    k2[inds_positive] = 1 - alpha2[inds_positive] / beta2[inds_positive]
    E_k2 = scipy.special.ellipe(k2)
    K_k2 = scipy.special.ellipkm1(1 - k2)  # more efficient than ellipk(k2), note the different argument
    # K_k2 = scipy.special.ellipk(k2)

    denom_rho = 2.0 * alpha2 * beta * rho
    with np.errstate(invalid='ignore'):
        numer_rho = C * Z * ((a ** 2 + rho ** 2 + Z ** 2) * E_k2 - alpha2 * K_k2)
    B_rho = numer_rho / denom_rho

    denom_z = 2.0 * alpha2 * beta
    with np.errstate(invalid='ignore'):
        numer_z = C * ((a ** 2 - rho ** 2 - Z ** 2) * E_k2 + alpha2 * K_k2)
    B_z = numer_z / denom_z

    return B_rho, B_z


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
    Algorithm based on "Volume-preserving algorithms for charged particle dynamics"
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
    omega_half = - q * B_norm / m
    v_plus = np.dot(expm(dt * omega_half * b_half_tensor), v_minus)
    v_new = v_plus + dt * q / m / 2.0 * E_half
    x_new = x_half + dt / 2.0 * v_new
    return x_new, v_new


def get_radius(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)
