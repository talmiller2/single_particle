import numpy as np

from em_fields.magnetic_forms import get_mirror_magnetic_field_z_component, get_transverse_magnetic_fields, get_radius


def smooth_step_function(z, z_step, dz_step, sigma_cutoff=6):
    """
    smooth step function that is 0 for z<z_step and 1 for z>step, with a smoothing length dz_step
    """
    if -(z - z_step) / dz_step > sigma_cutoff:
        W, dW_dz = 0, 0
    elif -(z - z_step) / dz_step < -sigma_cutoff:
        W, dW_dz = 1, 0
    else:
        W = 1 / (1 + np.exp(-(z - z_step) / dz_step))
        dW_dz = np.exp(-(z - z_step) / dz_step) / dz_step / (1 + np.exp(-(z - z_step) / dz_step)) ** 2
    return W, dW_dz


def get_main_cell_static_field(x, sigma_cutoff=6, **field_dict):
    z = x[2]
    Bz_static, dB_dz_static = 0, 0
    B_max = field_dict['B0'] * (field_dict['Rm'] - 1)
    dz_main = field_dict['MMM_static_main_cell_dz']
    for sign in [+1, -1]:
        zi = sign * field_dict['MMM_static_main_cell_z']
        if abs(z - zi) < sigma_cutoff * dz_main:
            Bz_static += B_max * np.exp(-((z - zi) / dz_main) ** 2)
            dB_dz_static += B_max * (-2 * (z - zi) / dz_main ** 2) * np.exp(-((z - zi) / dz_main) ** 2)
    return Bz_static, dB_dz_static

def get_MMM_magnetic_field(x, t, **field_dict):
    """
    Returns the magnetic field of the moving-multiple-mirror (MMM), all components x,y,z.
    """
    Bz_MMM = field_dict['B0']
    dBz_MMM_dz = 0

    if field_dict['use_static_main_cell'] == True:
        Bz_static, dB_dz_static = get_main_cell_static_field(x, **field_dict)
        Bz_MMM += Bz_static
        dBz_MMM_dz += dB_dz_static

    for sign in [+1, -1]:
        # moving field component
        x_m = [x[0], x[1], x[2] + sign * field_dict['U_MMM'] * t]
        B_mz, dB_mz_dz = get_mirror_magnetic_field_z_component(x_m, field_dict)
        B_mz -= field_dict['B0']

        # static main-cell-wall field component
        W, dW_dz = smooth_step_function(x[2], sign * field_dict['MMM_z_wall'], sign * field_dict['MMM_dz_wall'])

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
            W, _ = smooth_step_function(x[2], sign * field_dict['MMM_z_wall'], sign * field_dict['MMM_dz_wall'])

            # total E field magnitude
            E_MMM_magnitude += - r / 2 * sign * field_dict['U_MMM'] * dB_mz_dz * W

        theta_vec = np.array([- x[1] / r, x[0] / r, 0])
        E = field_dict['induced_fields_factor'] * E_MMM_magnitude * theta_vec
        return E
