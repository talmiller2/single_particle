import numpy as np

from em_fields.magnetic_forms import get_mirror_magnetic_field


def E_RF_function(x, t, **field_dict):
    """
    Electric field of planar RF wave in the z direction.
    """
    z = x[2]
    E_RF = field_dict['E_RF']
    anticlockwise = field_dict['anticlockwise']
    z_0 = field_dict['z_0']
    phase_RF = field_dict['phase_RF']
    E_RF_vector = 0
    for k, omega in zip(field_dict['k'], field_dict['omega']):
        E_RF_vector += E_RF * np.array([anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
                                        np.cos(k * (z - z_0) - omega * t + phase_RF),
                                        0])
    return E_RF_vector


def B_RF_function(x, t, **field_dict):
    """
    Magnetic field of a magnetic mirror +  planar RF wave in the z direction.
    """
    B0 = field_dict['B0']
    Rm = field_dict['Rm']
    l = field_dict['l']
    mirror_field_type = field_dict['mirror_field_type']
    B_mirror = get_mirror_magnetic_field(x, B0, Rm, l, mirror_field_type=mirror_field_type)

    # B_RF = 1/c * k_hat cross E_RF
    # https://en.wikipedia.org/wiki/Sinusoidal_plane-wave_solutions_of_the_electromagnetic_wave_equation
    # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
    z = x[2]
    E_RF = field_dict['E_RF']
    anticlockwise = field_dict['anticlockwise']
    z_0 = field_dict['z_0']
    phase_RF = field_dict['phase_RF']
    c = field_dict['c']
    B_RF_vector = 0
    for k, omega in zip(field_dict['k'], field_dict['omega']):
        B_RF_vector += E_RF / c * np.array([-np.sign(k) * np.cos(k * (z - z_0) - omega * t + phase_RF),
                                            np.sign(k) * anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
                                            0])
    return B_mirror + B_RF_vector
