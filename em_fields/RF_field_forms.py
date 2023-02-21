import numpy as np

from em_fields.magnetic_forms import get_mirror_magnetic_field


def E_RF_function(x_vec, t, **field_dict):
    """
    Electric field of planar RF wave in the z direction.
    """

    # choose RF where the electric or magnetic fields are transverse
    x = x_vec[0]
    y = x_vec[1]
    z = x_vec[2]
    anticlockwise = field_dict['anticlockwise']
    z_0 = field_dict['z_0']
    c = field_dict['c']

    E_RF_vector = 0
    if field_dict['RF_type'] == 'electric_transverse':
        E_RF = field_dict['E_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            E_RF_vector += E_RF * np.array([anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
                                            np.cos(k * (z - z_0) - omega * t + phase_RF),
                                            0])
    elif field_dict['RF_type'] == 'magnetic_transverse' and field_dict['use_RF_correction'] == True:
        B_RF = field_dict['B_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            Ez = - x * np.sin(k * (z - z_0) - omega * t + phase_RF) \
                 - y * anticlockwise * np.cos(k * (z - z_0) - omega * t + phase_RF)
            dEdz = - x * k * np.cos(k * (z - z_0) - omega * t + phase_RF) \
                   + y * k * anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF)
            E_RF_vector += B_RF * omega / c ** 2 * np.array([-x / 2 * dEdz, -y / 2 * dEdz, Ez])

    return E_RF_vector


def B_RF_function(x_vec, t, **field_dict):
    """
    Magnetic field of a magnetic mirror + RF
    """
    B0 = field_dict['B0']
    Rm = field_dict['Rm']
    l = field_dict['l']
    mirror_field_type = field_dict['mirror_field_type']
    B_mirror = get_mirror_magnetic_field(x_vec, B0, Rm, l, mirror_field_type=mirror_field_type)

    # choose RF where the electric or magnetic fields are transverse
    x = x_vec[0]
    y = x_vec[1]
    z = x_vec[2]
    anticlockwise = field_dict['anticlockwise']
    z_0 = field_dict['z_0']
    c = field_dict['c']

    B_RF_vector = 0
    if field_dict['RF_type'] == 'electric_transverse' and field_dict['use_RF_correction'] == True:
        E_RF = field_dict['E_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            Bz = - x * np.sin(k * (z - z_0) - omega * t + phase_RF) \
                 - y * anticlockwise * np.cos(k * (z - z_0) - omega * t + phase_RF)
            dBdz = - x * k * np.cos(k * (z - z_0) - omega * t + phase_RF) \
                   + y * k * anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF)
            B_RF_vector += E_RF * omega / c ** 2 * np.array([-x / 2 * dBdz, -y / 2 * dBdz, Bz])
    elif field_dict['RF_type'] == 'magnetic_transverse':
        B_RF = field_dict['B_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            B_RF_vector += B_RF * np.array([anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
                                            np.cos(k * (z - z_0) - omega * t + phase_RF),
                                            0])

    return B_mirror + B_RF_vector
