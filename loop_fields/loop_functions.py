import numpy as np
import scipy.special

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

