import numpy as np

from em_fields.em_functions import get_thermal_velocity, get_cyclotron_angular_frequency


def define_default_settings(settings=None):
    if settings == None:
        settings = {}

    # physical constants
    settings['eV'] = 1.0
    settings['keV'] = 1e3 * settings['eV']
    settings['eV_to_K'] = 1.16e4
    settings['MeV_to_J'] = 1e6 * 1.6e-19
    settings['kB_K'] = 1.380649e-23  # J/K
    settings['e'] = 1.60217662e-19  # Coulomb (elementary charge)
    settings['kB_eV'] = settings['kB_K'] * settings['eV_to_K']  # J/eV (numerically same as e)
    settings['eps0'] = 8.85418781e-12  # Farad/m^2 (vacuum permittivity)
    settings['c'] = 3e8  # m/s

    # plasma parameters5
    if 'gas_name' not in settings:
        settings['gas_name'] = 'hydrogen'
    if 'ionization_level' not in settings:
        settings['ionization_level'] = 1.0
    settings['me'], settings['mp'], settings['mi'], settings['A_atomic_weight'], settings['Z_ion'] \
        = define_plasma_parameters(gas_name=settings['gas_name'], ionization_level=settings['ionization_level'])
    settings['q'] = settings['Z_ion'] * settings['e']  # Coulomb

    # system parameters
    if 'T_keV' not in settings:
        settings['T_keV'] = 3.0
    settings['T_eV'] = settings['T_keV'] * 1e3
    settings['v_th'] = get_thermal_velocity(settings['T_eV'], settings['mi'], settings['kB_eV'])
    if 'Rm' not in settings:
        settings['Rm'] = 3.0
    settings['loss_cone_angle'] = np.arcsin(settings['Rm'] ** (-0.5)) * 360 / (2 * np.pi)

    settings['l'] = 10.0  # m (MM cell size)
    settings['r_0'] = 0.0 * settings['l']
    settings['z_0'] = 0.5 * settings['l']

    return settings


def define_default_field(settings, field_dict=None):
    if field_dict == None:
        field_dict = {}

    if 'B0' not in field_dict:
        field_dict['B0'] = 0.1  # Tesla
    field_dict['omega_cyclotron'] = get_cyclotron_angular_frequency(settings['q'], field_dict['B0'], settings['mi'])
    field_dict['tau_cyclotron'] = 2 * np.pi / field_dict['omega_cyclotron']

    if 'E_RF_kVm' not in field_dict:
        field_dict['E_RF_kVm'] = 0  # kV/m
    field_dict['E_RF'] = field_dict['E_RF_kVm'] * 1e3  # the SI units is V/m

    if field_dict['B0'] == 0:  # pick a default
        field_dict['anticlockwise'] = 1
    else:
        field_dict['anticlockwise'] = np.sign(field_dict['B0'])

    # field_dict['RF_type'] = 'uniform'
    field_dict['RF_type'] = 'traveling'
    if field_dict['RF_type'] == 'uniform':
        omega_RF = field_dict['omega_cyclotron']  # resonance
        k_RF = omega_RF / settings['c']
    elif field_dict['RF_type'] == 'traveling':
        # field_dict['alpha_detune_list'] = [2]
        field_dict['alpha_detune_list'] = [2.718]
        # field_dict['alpha_detune_list'] = [2, 2.718]
        # field_dict['alpha_detune_list'] = [2.718, 3.141]
        omega_RF = []
        v_RF = []
        k_RF = []
        for alpha_detune in field_dict['alpha_detune_list']:
            omega_RF += [alpha_detune * field_dict['omega_cyclotron']]  # resonance
            v_RF += [alpha_detune / (alpha_detune - 1) * settings['v_th']]
            k_RF += [omega_RF[-1] / v_RF[-1]]

    field_dict['l'] = settings['l']
    field_dict['l'] = settings['l']
    field_dict['z_0'] = settings['z_0']
    field_dict['c'] = settings['c']
    field_dict['mirror_field_type'] = 'logan'
    # field_dict['mirror_field_type'] = 'const'
    field_dict['phase_RF'] = 0
    field_dict['omega_RF'] = omega_RF
    field_dict['k_RF'] = k_RF

    return field_dict


def define_plasma_parameters(gas_name='hydrogen', ionization_level=1):
    me = 9.10938356e-31  # kg
    mp = 1.67262192e-27  # kg
    if gas_name == 'hydrogen':
        A = 1.00784
        Z = 1.0
    elif gas_name == 'deuterium':
        A = 2.01410177811
        Z = 1.0
    elif gas_name == 'tritium':
        A = 3.0160492
        Z = 1.0
    elif gas_name == 'DT_mix':
        A = np.mean([2.01410177811, 3.0160492])  # approximate as mean of D and T
        Z = 1.0
    elif gas_name == 'helium':
        A = 4.002602
        Z = 2.0
    elif gas_name == 'lithium':
        A = 6.941  # 92.41% Li7 A=7.016, 7.59% Li6 A=6.015 (Wikipedia)
        Z = 3.0
    elif gas_name == 'sodium':
        A = 22.9897
        Z = 11.0
    elif gas_name == 'potassium':
        A = 39.0983
        Z = 19.0
    else:
        raise TypeError('invalid gas: ' + gas_name)
    mi = A * mp
    # for non-fusion experiments with low temperature, the ions are not fully ionized
    if ionization_level is not None:
        Z = ionization_level
    return me, mp, mi, A, Z
