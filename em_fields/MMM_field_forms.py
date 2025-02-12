import numpy as np

from em_fields.magnetic_forms import get_mirror_magnetic_field_z_component, get_transverse_magnetic_fields, get_radius


def get_mail_cell_wall_function(x, **field_dict):
    z = x[2]
    z_wall = field_dict['MMM_z_wall']
    dz_wall = field_dict['MMM_dz_wall']
    W = 1 / (1 + np.exp(- (z - z_wall) / dz_wall))
    dW_dz = np.exp(- (z - z_wall) / dz_wall) / dz_wall / (1 + np.exp(- (z - z_wall) / dz_wall)) ** 2
    return W, dW_dz


def get_MMM_magnetic_field(x, t, **field_dict):
    """
    Returns the magnetic field of the moving-multiple-mirror (MMM), all components x,y,z.
    """
    Bz_MMM = field_dict['B0']
    dBz_MMM_dz = 0

    for sign in [+1, -1]:
        # moving field component
        x_m = [x[0], x[1], x[2] + sign * field_dict['U_MMM'] * t]
        B_mz, dB_mz_dz = get_mirror_magnetic_field_z_component(x_m, field_dict)
        B_mz -= field_dict['B0']

        # static main-cell-wall field component
        x_s = [x[0], x[1], sign * x[2]]
        W, dW_dz = get_mail_cell_wall_function(x_s, **field_dict)

        # total z component field
        B_z = B_mz * W
        dBz_dz = B_mz * dW_dz + dB_mz_dz * W

        Bz_MMM += B_z
        dBz_MMM_dz += dBz_dz

    if field_dict['use_transverse_fields'] == True:
        Bx, By = get_transverse_magnetic_fields(x, dBz_MMM_dz)
    else:
        Bx, By = 0, 0

    return np.array([Bx, By, Bz_MMM])


def get_MMM_electric_field(x, t, **field_dict):
    """
    Returns the electric field of the moving-multiple-mirror (MMM), all components x,y,z.
    """
    r = get_radius(x)
    if r == 0:
        return np.array([0, 0, 0])
    else:
        E_MMM_magnitude = 0
        for sign in [+1, -1]:
            # moving field component
            x_m = [x[0], x[1], x[2] + sign * field_dict['U_MMM'] * t]
            _, dB_mz_dz = get_mirror_magnetic_field_z_component(x_m, field_dict)

            # static main-cell-wall field component
            x_s = [x[0], x[1], sign * x[2]]
            W, _ = get_mail_cell_wall_function(x_s, **field_dict)

            # total E field magnitude
            E_MMM_magnitude += - r / 2 * sign * field_dict['U_MMM'] * dB_mz_dz * W

        theta_vec = np.array([- x[1] / r, x[0] / r, 0])
        E = field_dict['induced_fields_factor'] * E_MMM_magnitude * theta_vec
        return E
