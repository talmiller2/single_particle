import numpy as np

from em_fields.magnetic_forms import get_mirror_magnetic_field


def E_RF_function(x_vec, t, **field_dict):
    """
    Electric field of RF
    """

    # choose RF where the electric or magnetic fields are transverse
    x = x_vec[0]
    y = x_vec[1]
    z = x_vec[2]
    clockwise = field_dict['clockwise']
    z_0 = field_dict['z_0']
    ind_fac = field_dict['induced_fields_factor']

    E_RF_vector = 0

    if field_dict['RF_type'] == 'electric_transverse':
        E_RF = field_dict['E_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            phase = k * (z - z_0) - omega * t + phase_RF
            E_RF_vector += E_RF * np.array([np.cos(phase), clockwise * np.sin(phase), 0])
            if field_dict['with_kr_correction']:
                E_RF_vector += E_RF * k * np.array([0, 0, clockwise * y * np.cos(phase) - x * np.sin(phase)])

    elif field_dict['RF_type'] == 'magnetic_transverse':
        B_RF = field_dict['B_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            phase = k * (z - z_0) - omega * t + phase_RF
            E_amp = - ind_fac * B_RF * omega
            E_RF_vector += E_amp * np.array([0, 0, clockwise * x * np.cos(phase) + y * np.sin(phase)])
            if field_dict['with_kr_correction']:
                E_RF_vector += - E_amp * k * x * y * np.array([np.cos(phase), - clockwise * np.sin(phase), 0])

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
    ind_fac = field_dict['induced_fields_factor']
    z_0 = field_dict['z_0']
    c = field_dict['c']

    B_RF_vector = 0

    if field_dict['RF_type'] == 'electric_transverse':
        E_RF = field_dict['E_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            phase = k * (z - z_0) - omega * t + phase_RF
            B_amp = ind_fac * E_RF * omega / c ** 2
            B_RF_vector += B_amp * np.array([0, 0, clockwise * x * np.cos(phase) + y * np.sin(phase)])
            if field_dict['with_kr_correction']:
                B_RF_vector += - B_amp * k * x * y * np.array([np.cos(phase), - clockwise * np.sin(phase), 0])

    elif field_dict['RF_type'] == 'magnetic_transverse':
        B_RF = field_dict['B_RF']
        for k, omega, phase_RF in zip(field_dict['k_RF'], field_dict['omega_RF'], field_dict['phase_RF']):
            phase = k * (z - z_0) - omega * t + phase_RF
            B_RF_vector += B_RF * np.array([np.cos(phase), clockwise * np.sin(phase), 0])
            if field_dict['with_kr_correction']:
                B_RF_vector += B_RF * k * np.array([0, 0, clockwise * y * np.cos(phase) - x * np.sin(phase)])

    return B_mirror + B_RF_vector
