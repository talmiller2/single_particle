import numpy as np


def get_radius(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)


def get_transverse_magnetic_fields(x, dBz_dz):
    """
    Based on Maxwell equation div(B)=0
    https://www.tcd.ie/Physics/people/Peter.Gallagher/lectures/PlasmaPhysics/Lecture5_single_particle.pdf
    """
    # r = get_radius(x)
    # Br = -0.5 * r * dBz_dz
    # Bx = x[0] / r * Br
    # By = x[1] / r * Br
    Bx = -0.5 * x[0] * dBz_dz
    By = -0.5 * x[1] * dBz_dz
    return Bx, By


def magnetic_field_constant(B0):
    Bz = B0
    return np.array([0, 0, Bz])


def magnetic_field_linear(x, B0, l, use_transverse_fields=True, z0=0):
    """
    Linear profile for testing
    """
    z = x[2] - z0
    dBz_dz = B0 / (1 * l)
    Bz = B0 + dBz_dz * (z - l / 2)
    if use_transverse_fields:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])


def magnetic_field_logan(x, B0, Rm, l, use_transverse_fields=True, z0=0):
    """
    Magnetic field from Logan et al (1972)
    """
    z = x[2] - z0
    Bz = B0 * (1 + (Rm - 1.0) * np.sin(np.pi * (z - l / 2.0) / l) ** 2.0)
    dBz_dz = 2 * np.pi / l * B0 * (Rm - 1.0) * np.cos(np.pi * (z - l / 2.0) / l) \
             * np.sin(np.pi * (z - l / 2.0) / l)
    if use_transverse_fields:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])


def magnetic_field_jaeger(x, B0, Rm, l, use_transverse_fields=True, z0=0):
    """
    Magnetic field based on Jaeger et al (1972), actually same as Logan via trigonometric identities
    """
    z = x[2] - z0
    Bz = B0 * (1 + (Rm - 1) / 2.0 * (1 - np.cos(2 * np.pi * (z - l / 2) / l)))
    dBz_dz = np.pi / l * B0 * (Rm - 1) * np.sin(2 * np.pi * (z - l / 2) / l)
    if use_transverse_fields:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])


def magnetic_field_post(x, B0, Rm, l, use_transverse_fields=True, z0=0):
    """
    Magnetic field from Logan et al (1972), describing the more localized form of Post (1967)
    """
    z = x[2] - z0
    lamda = 5.5
    Bz = B0 * (1 + (Rm - 1.0) * np.exp(- lamda * np.sin(np.pi * z / l) ** 2.0))
    dBz_dz = - lamda * 2 * np.pi / l * B0 * (Rm - 1.0) * np.cos(np.pi * z / l) \
             * np.sin(np.pi * z / l) \
             * np.exp(- lamda * np.sin(np.pi * z / l) ** 2.0)
    if use_transverse_fields:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])
