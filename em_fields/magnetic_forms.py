import numpy as np


def get_mirror_magnetic_field(x, field_dict):
    """
    Single function that calls the various forms of mirror field: logan, jaeger or post.
    """

    # define defaults
    if 'mirror_field_type' not in field_dict:
        field_dict['mirror_field_type'] = 'post'
    if 'use_transverse_fields' not in field_dict:
        field_dict['use_transverse_fields'] = True
    if 'z0' not in field_dict:
        field_dict['z0'] = 0
    if 'B0' not in field_dict:
        field_dict['B0'] = 1  # [T]
    if 'Rm' not in field_dict:
        field_dict['Rm'] = 3
    if 'l' not in field_dict:
        field_dict['l'] = 1  # [m]

    if field_dict['mirror_field_type'] == 'logan':
        B_mirror = magnetic_field_logan(x, field_dict)
    elif field_dict['mirror_field_type'] == 'jaeger':
        B_mirror = magnetic_field_jaeger(x, field_dict)
    elif field_dict['mirror_field_type'] == 'post':
        B_mirror = magnetic_field_post(x, field_dict)
    elif field_dict['mirror_field_type'] == 'const':
        B_mirror = np.array([0, 0, field_dict['B0']])
    else:
        raise TypeError('invalid mirror_field_type: ' + str(field_dict['mirror_field_type']))

    if field_dict['use_mirror_slope']:
        if 'B_slope_fac' not in field_dict:
            field_dict['B_slope_fac'] = 1.0
        if 'B_slope_smooth_length' not in field_dict:
            field_dict['B_slope_smooth_length'] = 0.2
        B_slope = magnetic_field_slope(x, field_dict)
    else:
        B_slope = 0

    return B_mirror + B_slope


def get_radius(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)


def get_transverse_magnetic_fields(x, dBz_dz):
    """
    Based on Maxwell equation div(B)=0, paraxial approximation
    See 2008 - Fisch et al - Simulation of alpha-channeling in mirror machines (https://doi.org/10.1063/1.2903900)
    """
    Bx = -0.5 * x[0] * dBz_dz
    By = -0.5 * x[1] * dBz_dz
    return Bx, By


def magnetic_field_linear(x, field_dict):
    """
    Linear profile for testing
    """
    z0 = field_dict['z0']
    B0 = field_dict['B0']
    l = field_dict['l']
    z = x[2] - z0
    dBz_dz = B0 / (1 * l)
    Bz = B0 + dBz_dz * (z - l / 2)
    if field_dict['use_transverse_fields'] == True:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])


def magnetic_field_logan(x, field_dict):
    """
    Magnetic field from Logan et al (1972)
    """
    z0 = field_dict['z0']
    B0 = field_dict['B0']
    Rm = field_dict['Rm']
    l = field_dict['l']
    z = x[2] - z0
    Bz = B0 * (1 + (Rm - 1.0) * np.sin(np.pi * (z - l / 2.0) / l) ** 2.0)
    dBz_dz = 2 * np.pi / l * B0 * (Rm - 1.0) * np.cos(np.pi * (z - l / 2.0) / l) \
             * np.sin(np.pi * (z - l / 2.0) / l)
    if field_dict['use_transverse_fields'] == True:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])


def magnetic_field_jaeger(x, field_dict):
    """
    Magnetic field based on Jaeger et al (1972), actually same as Logan via trigonometric identities
    """
    z0 = field_dict['z0']
    B0 = field_dict['B0']
    Rm = field_dict['Rm']
    l = field_dict['l']
    z = x[2] - z0
    Bz = B0 * (1 + (Rm - 1) / 2.0 * (1 - np.cos(2 * np.pi * (z - l / 2) / l)))
    dBz_dz = np.pi / l * B0 * (Rm - 1) * np.sin(2 * np.pi * (z - l / 2) / l)
    if field_dict['use_transverse_fields'] == True:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])


def magnetic_field_post(x, field_dict):
    """
    Magnetic field from Logan et al (1972), describing the more localized form of Post (1967)
    """
    z0 = field_dict['z0']
    B0 = field_dict['B0']
    Rm = field_dict['Rm']
    l = field_dict['l']
    if 'lambda_post' not in field_dict:
        field_dict['lambda_post'] = 5.5
    lamda = field_dict['lambda_post']
    z = x[2] - z0
    Bz = B0 * (1 + (Rm - 1.0) * np.exp(- lamda * np.sin(np.pi * z / l) ** 2.0))
    dBz_dz = - lamda * 2 * np.pi / l * B0 * (Rm - 1.0) * np.cos(np.pi * z / l) \
             * np.sin(np.pi * z / l) \
             * np.exp(- lamda * np.sin(np.pi * z / l) ** 2.0)
    if field_dict['use_transverse_fields'] == True:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0
    return np.array([Bx, By, Bz])


def magnetic_field_slope(x, field_dict):
    """
    Magnetic field slope
    """
    z0 = field_dict['z0']
    l = field_dict['l']
    B_s = field_dict['B_slope']
    B_slope_smooth_length = field_dict['B_slope_smooth_length']
    z = x[2] - z0
    z_mod = np.mod(z, l)
    sigma_s = l * B_slope_smooth_length
    Bz = B_s * (z_mod - l / 2.0) / l \
         * (1 - np.exp(- (z_mod - l) ** 2.0 / sigma_s ** 2.0)) \
         * (1 - np.exp(- z_mod ** 2.0 / sigma_s ** 2.0))
    dBz_dz = (B_s / l * np.exp(- z_mod ** 2.0 / sigma_s ** 2.0)
              * (1 - np.exp(- (z_mod - l) ** 2.0 / sigma_s ** 2.0)))
    dBz_dz += (B_s * 2 * z_mod * (z_mod - l / 2.0) / l / sigma_s ** 2
               * (np.exp(- z_mod ** 2.0 / sigma_s ** 2.0)
                  * (1 - np.exp(- (z_mod - l) ** 2.0 / sigma_s ** 2.0))))
    dBz_dz += (B_s * 2 * (z_mod - l / 2.0) * (z_mod - l) / l / sigma_s ** 2
               * (np.exp(- (z_mod - l) ** 2.0 / sigma_s ** 2.0)
                  * (1 - np.exp(- z_mod ** 2.0 / sigma_s ** 2.0))))

    if field_dict['use_transverse_fields'] == True:
        Bx, By = get_transverse_magnetic_fields(x, dBz_dz)
    else:
        Bx, By = 0, 0

    return np.array([Bx, By, Bz])
