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

    # plasma parameters
    if 'gas_name' not in settings:
        settings['gas_name'] = 'hydrogen'
    if 'gas_name_for_cyc' not in settings:
        settings['gas_name_for_cyc'] = 'hydrogen'
    if 'ionization_level' not in settings:
        settings['ionization_level'] = 1.0
    settings['me'], settings['mp'], settings['mi'], settings['A_atomic_weight'], settings['Z_ion'] \
        = define_plasma_parameters(gas_name=settings['gas_name'], ionization_level=settings['ionization_level'])
    settings['q'] = settings['Z_ion'] * settings['e']  # Coulomb
    _, _, settings['mi_for_cyc'], _, settings['Z_ion_for_cyc'] \
        = define_plasma_parameters(gas_name=settings['gas_name_for_cyc'], ionization_level=settings['ionization_level'])
    settings['q_for_cyc'] = settings['Z_ion_for_cyc'] * settings['e']  # Coulomb

    # system parameters
    if 'T_keV' not in settings:
        # settings['T_keV'] = 3.0
        settings['T_keV'] = 10.0
    settings['T_eV'] = settings['T_keV'] * 1e3
    settings['v_th'] = get_thermal_velocity(settings['T_eV'], settings['mi'], settings['kB_eV'])
    settings['v_th_for_cyc'] = get_thermal_velocity(settings['T_eV'], settings['mi_for_cyc'], settings['kB_eV'])
    if 'l' not in settings:
        settings['l'] = 1.0  # m (MM cell size)
    if 'z_0' not in settings:
        settings['z_0'] = 0.5 * settings['l']
    if 'radial_distribution' not in settings:
        settings['radial_distribution'] = 'uniform'
    if 'sigma_r0' not in settings:
        settings['sigma_r0'] = 0

    # simulation parameters
    if 'time_step_tau_cyclotron_divisions' not in settings:
        settings['time_step_tau_cyclotron_divisions'] = 20.0
    if 'trajectory_save_method' not in settings:
        settings['trajectory_save_method'] = 'intervals'
        # settings['trajectory_save_method'] = 'min_Bz'
        # settings['trajectory_save_method'] = 'min_Bz_mirror_const_vz_sign'
    if 'stop_criterion' not in settings:
        settings['stop_criterion'] = 'steps'
        # settings['stop_criterion'] = 't_max'
        # settings['stop_criterion'] = 't_max_adaptive_dt'
        # settings['stop_criterion'] = 'first_cell_center_crossing'
        # settings['stop_criterion'] = 'several_cell_center_crossing'
    if 'r_max' not in settings:
        settings['r_max'] = None
    if 'number_of_time_intervals' not in settings:
        settings['number_of_time_intervals'] = 1
    if 'num_snapshots' not in settings:
        settings['num_snapshots'] = 300
    if 'set_save_format' not in settings:
        # settings['set_save_format'] = 'mat'
        settings['set_save_format'] = 'pickle'
    if 'absolute_velocity_sampling_type' not in settings:
        # settings['absolute_velocity_sampling_type'] = 'const_vth'
        settings['absolute_velocity_sampling_type'] = 'maxwell'
    if 'direction_velocity_sampling_type' not in settings:
        # settings['direction_velocity_sampling_type'] = 'right_loss_cone'
        settings['direction_velocity_sampling_type'] = '4pi'
    if 'apply_random_RF_phase' not in settings:
        # settings['apply_random_RF_phase'] = False
        settings['apply_random_RF_phase'] = True

    return settings


def define_default_field(settings, field_dict=None):
    if field_dict == None:
        field_dict = {}

    # single mirror properties
    if 'mirror_field_type' not in field_dict:
        # field_dict['mirror_field_type'] = 'logan'
        field_dict['mirror_field_type'] = 'post'
        # field_dict['mirror_field_type'] = 'const'
    if 'Rm' not in field_dict:
        field_dict['Rm'] = 3.0  # mirror ratio
    field_dict['loss_cone_angle'] = np.arcsin(field_dict['Rm'] ** (-0.5)) * 360 / (2 * np.pi)
    if 'B0' not in field_dict:
        # field_dict['B0'] = 0.1  # Tesla
        field_dict['B0'] = 1.0  # Tesla
    field_dict['omega_cyclotron'] = get_cyclotron_angular_frequency(settings['q_for_cyc'], field_dict['B0'],
                                                                    settings['mi_for_cyc'])
    field_dict['tau_cyclotron'] = 2 * np.pi / field_dict['omega_cyclotron']
    if 'l' not in field_dict:
        field_dict['l'] = settings['l']
    if 'z_mirror_shift' not in field_dict:
        field_dict['z_mirror_shift'] = 0
    if 'use_transverse_fields' not in field_dict:
        field_dict['use_transverse_fields'] = True

    # mirror slope properties
    if 'use_mirror_slope' not in field_dict:
        field_dict['use_mirror_slope'] = False
    if 'B_slope' not in field_dict:
        field_dict['B_slope'] = 1.0  # [T]
    if 'B_slope_smooth_length' not in field_dict:
        field_dict['B_slope_smooth_length'] = 0.2

    # RF chirp properties
    if 'use_RF_chirp' not in field_dict:
        field_dict['use_RF_chirp'] = False
    if 'RF_chirp_period' not in field_dict:
        field_dict['RF_chirp_period'] = 1e3
    if 'RF_chirp_amplitude' not in field_dict:
        field_dict['RF_chirp_amplitude'] = 0.1
    if 'RF_chirp_time_offset' not in field_dict:
        field_dict['RF_chirp_time_offset'] = 0

    # RF properties
    if 'clockwise' not in field_dict:
        if field_dict['B0'] == 0:  # pick a default
            field_dict['clockwise'] = 1
        else:
            field_dict['clockwise'] = np.sign(field_dict['B0'])
    if 'RF_type' not in field_dict:
        field_dict['RF_type'] = 'electric_transverse'
        # field_dict['RF_type'] = 'magnetic_transverse'
    if 'E_RF_kVm' not in field_dict:
        field_dict['E_RF_kVm'] = 0  # kV/m
    field_dict['E_RF'] = field_dict['E_RF_kVm'] * 1e3  # the SI units is V/m
    if 'B_RF' not in field_dict:
        field_dict['B_RF'] = 1e-3  # the SI units is T
    if 'with_kr_correction' not in field_dict:
        field_dict['with_kr_correction'] = True
    if 'induced_fields_factor' not in field_dict:
        field_dict['induced_fields_factor'] = 1.0

    if 'alpha_RF_list' not in field_dict:
        field_dict['alpha_RF_list'] = [1.0]
    if 'beta_RF_list' not in field_dict:
        field_dict['beta_RF_list'] = [0]
    if 'phase_RF_addition' not in field_dict:
        field_dict['phase_RF_addition'] = 0

    omega_RF = []
    k_RF = []
    phase_RF = []
    cnt = 1
    for alpha_RF, beta_RF in zip(field_dict['alpha_RF_list'], field_dict['beta_RF_list']):
        omega_RF += [alpha_RF * field_dict['omega_cyclotron']]
        k_RF += [beta_RF * 2 * np.pi / field_dict['l']]
        # pull the different RF waves out of sync
        phase_RF += [np.pi * cnt ** (np.e - 1) / len(omega_RF) + field_dict['phase_RF_addition']]
        cnt += 1

    field_dict['c'] = settings['c']
    field_dict['omega_RF'] = omega_RF
    field_dict['k_RF'] = k_RF
    field_dict['phase_RF'] = phase_RF

    # MMM definitions
    if 'MMM_z_wall' not in field_dict:
        field_dict['MMM_z_wall'] = 1.0  # [m]
        # field_dict['MMM_z_wall'] = 1.2  # [m]
    if 'MMM_dz_wall' not in field_dict:
        field_dict['MMM_dz_wall'] = 0.05  # [m]
        # field_dict['MMM_dz_wall'] = 0.1  # [m]
    if 'Rm_main' not in field_dict:
        field_dict['Rm_main'] = field_dict['Rm']
    if 'use_static_main_cell' not in field_dict:
        field_dict['use_static_main_cell'] = True
    if 'MMM_static_main_cell_z' not in field_dict:
        field_dict['MMM_static_main_cell_z'] = 1.0  # [m]
    if 'MMM_static_main_cell_dz' not in field_dict:
        field_dict['MMM_static_main_cell_dz'] = 0.05  # [m]
        # field_dict['MMM_static_main_cell_dz'] = 0.1 # [m]

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
