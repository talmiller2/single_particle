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
    clockwise = field_dict['clockwise']
    z_0 = field_dict['z_0']
    c = field_dict['c']

    E_RF_vector = 0
    if field_dict['RF_type'] == 'electric_transverse':
        E_RF = field_dict['E_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            E_RF_vector += E_RF * np.array([np.cos(k * (z - z_0) - omega * t + phase_RF),
                                            clockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
                                            0])

    elif field_dict['RF_type'] == 'magnetic_transverse':
        B_RF = field_dict['B_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            Ez = -B_RF * omega * (clockwise * x * np.cos(k * (z - z_0) - omega * t + phase_RF)
                                  + y * np.sin(k * (z - z_0) - omega * t + phase_RF))
            E_RF_vector += field_dict['induced_fields_factor'] * np.array([0, 0, Ez])
            if field_dict['with_RF_xy_corrections']:
                dEdz = -B_RF * omega * (-clockwise * x * k * np.sin(k * (z - z_0) - omega * t + phase_RF)
                                        + y * k * np.cos(k * (z - z_0) - omega * t + phase_RF))
                E_RF_vector += field_dict['induced_fields_factor'] * np.array([-x / 2 * dEdz, -y / 2 * dEdz, 0])

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
    clockwise = field_dict['clockwise']
    z_0 = field_dict['z_0']
    c = field_dict['c']

    B_RF_vector = 0
    if field_dict['RF_type'] == 'electric_transverse':
        E_RF = field_dict['E_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            Bz = E_RF * omega / c ** 2 * (clockwise * x * np.cos(k * (z - z_0) - omega * t + phase_RF)
                                          + y * np.sin(k * (z - z_0) - omega * t + phase_RF))
            B_RF_vector += field_dict['induced_fields_factor'] * np.array([0, 0, Bz])
            if field_dict['with_RF_xy_corrections']:
                dBdz = E_RF * omega / c ** 2 * (-clockwise * x * k * np.sin(k * (z - z_0) - omega * t + phase_RF)
                                                + y * k * np.cos(k * (z - z_0) - omega * t + phase_RF))
                B_RF_vector += field_dict['induced_fields_factor'] * np.array([-x / 2 * dBdz, -y / 2 * dBdz, 0])

    elif field_dict['RF_type'] == 'magnetic_transverse':
        B_RF = field_dict['B_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            B_RF_vector += B_RF * np.array([np.cos(k * (z - z_0) - omega * t + phase_RF),
                                            clockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
                                            0])

    return B_mirror + B_RF_vector
