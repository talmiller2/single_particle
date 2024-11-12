import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import maxwell
from slurmpy.slurmpy import Slurm

from em_fields.default_settings import define_default_settings, define_default_field
from em_fields.slurm_functions import get_script_evolution_slave

evolution_slave_script = get_script_evolution_slave()

slurm_kwargs = {}
slurm_kwargs['partition'] = 'core'
# slurm_kwargs['partition'] = 'testCore'
# slurm_kwargs['partition'] = 'socket'
# slurm_kwargs['partition'] = 'testSocket'
slurm_kwargs['ntasks'] = 1
# slurm_kwargs['cpus-per-task'] = 1
# slurm_kwargs['cores-per-socket'] = 1

local = False
# local = True

save_dir = '/home/talm/code/single_particle/slurm_runs/'
# save_dir += '/set5/'
# save_dir += '/set6/'
# save_dir += '/set7_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir += '/set8_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir += '/set9_T_10keV_B0_1T_Rm_2_l_1_phase_pi/'
# save_dir += '/set10_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir += '/set11_T_B0_1T_Rm_2_l_1m_randphase/'
# save_dir += '/set12_T_B0_1T_Rm_4_l_1m_randphase/'
# save_dir += '/set13_T_B0_1T_Rm_2_l_1m_randphase/'
# save_dir += '/set14_T_B0_1T_l_1m_randphase_save_intervals/'
# save_dir += '/set15_T_B0_1T_l_1m_Logan_intervals/'
# save_dir += '/set16_T_B0_1T_l_1m_Post_intervals/'
# save_dir += '/set17_T_B0_1T_l_3m_Post_intervals/'
# save_dir += '/set18_T_B0_1T_l_3m_Logan_intervals/'
# save_dir += '/set19_T_B0_1T_l_3m_Post_intervals_Rm_1.3/'
# save_dir += '/set20_B0_1T_l_3m_Post_intervals_Rm_3/'
# save_dir += '/set21_B0_1T_l_3m_Post_intervals_Rm_3_different_phases/'
# save_dir += '/set22_B0_1T_l_3m_Post_intervals_Rm_3/'
# save_dir += '/set23_B0_1T_l_3m_Post_intervals_Rm_6/'
# save_dir += '/set24_B0_1T_l_3m_Post_Rm_3/'
# save_dir += '/set25_B0_1T_l_3m_Post_Rm_3/'
# save_dir += '/set26_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set28_B0_1T_l_10m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set29_B0_1T_l_3m_Post_Rm_2_first_cell_center_crossing/'
# save_dir += '/set30_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set31_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set32_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set33_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set34_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set35_B0_0.1T_l_1m_Post_Rm_5_intervals/'
# save_dir += '/set36_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set37_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set38_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set39_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set40_B0_1T_l_1m_Logan_Rm_3_intervals_D_T/'
# save_dir += '/set41_B0_1T_l_1m_Post_Rm_3_intervals_D_T_ERF_25/'
# save_dir += '/set42_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set43_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set44_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set45_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set46_B0_2T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set48_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set49_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
save_dir += '/set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'

plt.close('all')

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 15), 2) # set24
# lambda_RF_loop_list = np.round(np.linspace(-20, 20, 10), 0)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set25
# lambda_RF_loop_list = np.round(np.linspace(-6, 6, 10), 0)
# lambda_RF_loop_list += np.sign(lambda_RF_loop_list)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
# beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.6, 1.0, 21), 2)  # set28
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, 30
# beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 11), 2)  # set31, 32, 33
# beta_loop_list = np.round(np.linspace(-10, 0, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set34
# beta_loop_list = np.round(np.linspace(-5, 5, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 5), 2)  # set35
# beta_loop_list = np.round(np.linspace(-10, 0, 5), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.2, 21), 2)  # set36
# beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.5, 1.5, 21), 2)  # set37, 39, 40
# beta_loop_list = np.round(np.linspace(-10, 10, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 21), 2)  # set38
# beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)

# alpha_loop_list = [1, 1.4, 1, 0.7, 0.55]  # set42, select sets from 2023 paper
# beta_loop_list = [0, 3, -3, -3, -7]

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 11), 2)  # set43
# beta_loop_list = np.round(np.linspace(-2, 2, 11), 2)

alpha_loop_list = np.round(np.linspace(0.4, 1.6, 21), 2)  # set47, 49, 50
beta_loop_list = np.round(np.linspace(-2, 2, 21), 2)

# # specific values for set48
# select_alpha_list = []
# select_beta_list = []
# select_alpha_list += [0.64]  # [set1]
# select_beta_list += [-1.8]
# select_alpha_list += [0.7]  # [set2]
# select_beta_list += [-0.8]
# select_alpha_list += [1.06]  # [set3]
# select_beta_list += [-1.8]
# select_alpha_list += [1.12]  # [set4]
# select_beta_list += [1.4]
# select_alpha_list += [0.88]  # [set5]
# select_beta_list += [0.0]
# alpha_loop_list = select_alpha_list
# beta_loop_list = select_beta_list

# RF_type = 'electric_transverse'
# E_RF_kVm = 25  # kV/m
E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

RF_type = 'magnetic_transverse'
# B_RF = 0.02  # T
B_RF = 0.04  # T
# B_RF = 0.08  # T

use_RF = True
# use_RF = False
if use_RF is False:
    E_RF_kVm = 0
    B_RF = 0
    alpha_loop_list = [1]
    beta_loop_list = [0]

loop_method = 'matrix'
# loop_method = 'array'

# gas_name_list = ['deuterium', 'tritium']
gas_name_list = ['deuterium']
# sigma_r0_list = [0, 0.1]
# induced_fields_factor_list = [1, 0.5, 0]
# sigma_r0_list = [0.1]
sigma_r0_list = [0.05]
induced_fields_factor_list = [1, 0]
# induced_fields_factor_list = [1]

for gas_name in gas_name_list:
    for sigma_r0 in sigma_r0_list:
        for induced_fields_factor in induced_fields_factor_list:

            if loop_method == 'matrix':
                combinations_list = []
                for beta in beta_loop_list:
                    for alpha in alpha_loop_list:
                        combinations_list += [{'alpha': alpha, 'beta': beta}]
            else:
                combinations_list = []
                for beta, alpha in zip(beta_loop_list, alpha_loop_list):
                    combinations_list += [{'alpha': alpha, 'beta': beta}]

            totol_combinations = len(combinations_list)
            print('totol_combinations = ' + str(totol_combinations))

            # divide the points to a given number of cpus
            num_cpus = 1
            # num_cpus = 10
            # num_cpus = 50
            # num_cpus = 200

            cnt_combination = 1

            for combination in combinations_list:
                alpha = combination['alpha']
                beta = combination['beta']
                print('run combination #' + str(cnt_combination) + '/' + str(totol_combinations)
                      + ': alpha=' + str(alpha) + ', beta=' + str(beta))

                # define settings
                settings = {}

                settings['trajectory_save_method'] = 'intervals'
                # settings['stop_criterion'] = 'steps'
                settings['stop_criterion'] = 't_max_adaptive_dt'
                # settings['stop_criterion'] = 'first_cell_center_crossing'
                # settings['stop_criterion'] = 'several_cell_center_crossing'
                # settings['number_of_time_intervals'] = 3

                settings['l'] = 1.0  # m (MM cell size)
                # settings['l'] = 3.0  # m (MM cell size)
                # settings['l'] = 10.0  # m (MM cell size)

                # settings['absolute_velocity_sampling_type'] = 'const_vth'
                settings['absolute_velocity_sampling_type'] = 'maxwell'

                # settings['direction_velocity_sampling_type'] = 'deterministic'

                settings['T_keV'] = 10.0
                # settings['T_keV'] = 30.0 / 1e3
                # settings['T_keV'] = 60.0 / 1e3

                # settings['gas_name'] = 'deuterium'
                # settings['gas_name'] = 'DT_mix'
                # settings['gas_name'] = 'tritium'
                settings['gas_name'] = gas_name
                settings['gas_name_for_cyc'] = 'DT_mix'

                # settings['time_step_tau_cyclotron_divisions'] = 20
                # settings['time_step_tau_cyclotron_divisions'] = 40 # for set 48 and before
                # settings['time_step_tau_cyclotron_divisions'] = 80
                settings['time_step_tau_cyclotron_divisions'] = 50  # for set 49

                settings['z_0'] = 0.5 * settings['l']

                # settings['sigma_r0'] = 0
                # settings['sigma_r0'] = 0.1
                settings['sigma_r0'] = sigma_r0

                settings['r_max'] = settings['l']

                settings = define_default_settings(settings)

                field_dict = {}

                # field_dict['B0'] = 0.1  # Tesla (1000 Gauss)
                field_dict['B0'] = 1.0  # Tesla
                # field_dict['B0'] = 2.0  # Tesla

                # field_dict['Rm'] = 1.3  # mirror ratio
                # field_dict['Rm'] = 2.0  # mirror ratio
                field_dict['Rm'] = 3.0  # mirror ratio
                # field_dict['Rm'] = 4.0  # mirror ratio
                # field_dict['Rm'] = 5.0  # mirror ratio
                # field_dict['Rm'] = 10.0  # mirror ratio

                if RF_type == 'electric_transverse':
                    field_dict['RF_type'] = 'electric_transverse'
                    field_dict['E_RF_kVm'] = E_RF_kVm
                elif RF_type == 'magnetic_transverse':
                    field_dict['RF_type'] = 'magnetic_transverse'
                    field_dict['B_RF'] = B_RF

                # field_dict['phase_RF_addition'] = 0
                # field_dict['phase_RF_addition'] = np.pi

                field_dict['alpha_RF_list'] = [alpha]
                field_dict['beta_RF_list'] = [beta]

                field_dict['mirror_field_type'] = 'post'
                # field_dict['mirror_field_type'] = 'logan'

                # field_dict['induced_fields_factor'] = 1
                # field_dict['induced_fields_factor'] = 0.5
                # field_dict['induced_fields_factor'] = 0.1
                # field_dict['induced_fields_factor'] = 0.01
                # field_dict['induced_fields_factor'] = 0
                field_dict['induced_fields_factor'] = induced_fields_factor

                field_dict['with_kr_correction'] = True
                # field_dict['with_kr_correction'] = False

                field_dict = define_default_field(settings, field_dict)

                # simulation duration
                settings['num_snapshots'] = 30
                # settings['num_snapshots'] = 10 * 30  # for specific runs, set 48

                # tmax_mirror_lengths = 2
                # sim_cyclotron_periods = (tmax_mirror_lengths * settings['l']
                #                          / settings['v_th_for_cyc'] / field_dict['tau_cyclotron'])
                # settings['sim_cyclotron_periods'] = sim_cyclotron_periods
                # settings['t_max'] = settings['sim_cyclotron_periods'] * field_dict['tau_cyclotron']
                # settings['t_max'] = 2.2937178074285e-06 (sets 47 and before)
                # settings['t_max'] = 2.3e-05  # longer time for specific runs (set 48)
                settings['t_max'] = 5 * settings['l'] / settings[
                    'v_th']  # longer time that depends on D,T v_th (set 49)
                settings['dt'] = field_dict['tau_cyclotron'] / settings['time_step_tau_cyclotron_divisions']
                if settings['stop_criterion'] in ['t_max', 't_max_adaptive_dt']:
                    settings['num_steps'] = int(1e10)
                else:
                    settings['num_steps'] = int(settings['t_max'] / settings['dt'])

                run_name = ''
                # run_name += 'tmax_' + str(settings['sim_cyclotron_periods'])
                # run_name += '_B0_' + str(field_dict['B0'])
                # run_name += '_T_' + str(settings['T_keV'])
                # run_name += 'Rm_' + str(int(field_dict['Rm']))
                if use_RF is False:
                    run_name += 'without_RF'
                else:
                    if RF_type == 'electric_transverse':
                        run_name += 'ERF_' + str(field_dict['E_RF_kVm'])
                    elif RF_type == 'magnetic_transverse':
                        run_name += 'BRF_' + str(field_dict['B_RF'])
                    run_name += '_alpha_' + '_'.join([str(a) for a in field_dict['alpha_RF_list']])
                    run_name += '_beta_' + '_'.join([str(b) for b in field_dict['beta_RF_list']])
                    if field_dict['induced_fields_factor'] < 1.0:
                        run_name += '_iff' + str(field_dict['induced_fields_factor'])
                    if field_dict['with_kr_correction'] == True:
                        run_name += '_withkrcor'
                run_name += '_tcycdivs' + str(settings['time_step_tau_cyclotron_divisions'])
                if settings['absolute_velocity_sampling_type'] == 'const_vth':
                    run_name += '_const_vth'
                if settings['sigma_r0'] > 0:
                    run_name += '_sigmar' + str(settings['sigma_r0'])
                    if settings['radial_distribution'] == 'normal':
                        run_name += 'norm'
                    elif settings['radial_distribution'] == 'uniform':
                        run_name += 'unif'

                if settings['gas_name'] != 'hydrogen':
                    run_name += '_' + settings['gas_name']

                print('run_name: ' + str(run_name))
                settings['run_name'] = run_name

                if num_cpus == 1:
                    settings['save_dir'] = save_dir
                else:
                    settings['save_dir'] = save_dir + '/' + run_name

                # total_number_of_points = 1
                # total_number_of_points = 40
                # total_number_of_points = 1000
                total_number_of_points = 2500
                # total_number_of_points = 3000
                # total_number_of_points = 5000

                # allow reproducibility
                np.random.seed(0)

                # initialize points data structure
                points_dict = {}

                # define absolute velocities of particles
                if settings['absolute_velocity_sampling_type'] == 'const_vth':
                    # using constant absolute velocity
                    v_abs_samples = settings['v_th'] * np.ones(total_number_of_points)
                elif settings['absolute_velocity_sampling_type'] == 'maxwell':
                    # sampling velocity from Maxwell-Boltzmann
                    scale = np.sqrt(settings['kB_eV'] * settings['T_eV'] / settings['mi'])
                    v_abs_samples = maxwell.rvs(size=total_number_of_points, scale=scale)
                else:
                    raise ValueError('invalid absolute_velocity_sampling_type :'
                                     + str(settings['absolute_velocity_sampling_type']))

                # define velocity directions of particles
                if settings['direction_velocity_sampling_type'] == '4pi':
                    # sampling a random 4 pi direction
                    rand_unit_vec = np.random.randn(total_number_of_points, 3)
                    for i in range(total_number_of_points):
                        rand_unit_vec[i, :] /= np.linalg.norm(rand_unit_vec[i, :])
                elif settings['direction_velocity_sampling_type'] == 'right_loss_cone':
                    # sampling a random direction but only within the right-LC
                    u = np.random.rand(total_number_of_points)
                    v = np.random.rand(total_number_of_points)
                    theta_max = settings['loss_cone_angle'] / 360 * 2 * np.pi
                    v_min = (np.cos(theta_max) + 1) / 2
                    v *= (1 - v_min)
                    v += v_min
                    phi = 2 * np.pi * u  # longitude
                    theta = np.arccos(2 * v - 1)  # latitude
                    x = np.cos(phi) * np.sin(theta)
                    y = np.sin(phi) * np.sin(theta)
                    z = np.cos(theta)
                    rand_unit_vec = np.array([x, y, z]).T
                elif settings['direction_velocity_sampling_type'] == 'deterministic':
                    loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
                    angles = []
                    angles += [loss_cone_angle * 1.1]
                    angles += [loss_cone_angle * 0.9]
                    angles += [180 - loss_cone_angle * 1.1]
                    angles += [180 - loss_cone_angle * 0.9]
                    x = []
                    y = []
                    z = []
                    size_subsamples = int(total_number_of_points / len(angles))
                    for i, angle in enumerate(angles):
                        x += [0 for _ in range(size_subsamples)]
                        y += [np.sin(angle / 360 * 2 * np.pi) for _ in range(size_subsamples)]
                        z += [np.cos(angle / 360 * 2 * np.pi) for _ in range(size_subsamples)]
                    rand_unit_vec = np.array([x, y, z]).T
                else:
                    raise ValueError('invalid direction_velocity_sampling_type :'
                                     + str(settings['direction_velocity_sampling_type']))

                # total velocity vector
                v_0 = rand_unit_vec
                for i in range(total_number_of_points):
                    v_0[i, :] *= v_abs_samples[i]
                points_dict['v_0'] = v_0

                # define initial positions of particles
                # sampling a random 2 pi direction
                rand_unit_vec = np.random.randn(total_number_of_points, 2)
                for i in range(total_number_of_points):
                    rand_unit_vec[i, :] /= np.linalg.norm(rand_unit_vec[i, :])
                if settings['radial_distribution'] == 'normal':
                    rand_r0_vec = abs(np.random.randn(total_number_of_points) * settings['sigma_r0'])
                elif settings['radial_distribution'] == 'uniform':
                    rand_r0_vec = abs(np.random.rand(total_number_of_points) * settings['sigma_r0'])

                x = rand_unit_vec[:, 0] * rand_r0_vec
                y = rand_unit_vec[:, 1] * rand_r0_vec
                z = settings['z_0'] + 0 * rand_r0_vec
                x_0 = np.array([x, y, z]).T
                points_dict['x_0'] = x_0

                # random RF phases for each particle
                if settings['apply_random_RF_phase']:
                    points_dict['phase_RF'] = 2 * np.pi * np.random.rand(total_number_of_points)

                # save the run settings for one example
                if cnt_combination == 1:
                    os.makedirs(settings['save_dir'], exist_ok=True)
                    os.chdir(settings['save_dir'])

                    settings_file = settings['save_dir'] + '/settings.pickle'
                    with open(settings_file, 'wb') as handle:
                        pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    field_dict_file = settings['save_dir'] + '/field_dict.pickle'
                    with open(field_dict_file, 'wb') as handle:
                        pickle.dump(field_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # points_dict_file = settings['save_dir'] + '/points_dict.mat'
                    # savemat(points_dict_file, points_dict)

                points_dict_file = settings['save_dir'] + '/points_dict_' + run_name + '.pickle'
                with open(points_dict_file, 'wb') as handle:
                    pickle.dump(points_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                num_points_per_cpu = int(np.floor(1.0 * total_number_of_points / num_cpus))
                num_extra_points = np.mod(total_number_of_points, num_cpus)
                points_set_list = []
                index_first = 0
                num_sets = num_cpus if num_points_per_cpu > 0 else num_extra_points
                for i in range(num_sets):
                    index_last = index_first + num_points_per_cpu
                    if i < num_extra_points:
                        index_last += 1
                    points_set_list += [[k for k in range(index_first, index_last)]]
                    index_first = index_last

                # run the slave scripts on multiple cpus
                cnt_cpu = 0
                for ind_set, points_set in enumerate(points_set_list):
                    settings['ind_set'] = ind_set
                    settings['points_set'] = points_set

                    if num_cpus == 1:
                        slurm_run_name = run_name
                        settings['ind_set'] = None
                    elif num_cpus > 1:
                        slurm_run_name = 'set_' + str(ind_set) + '_' + run_name
                    print('run_name = ' + run_name)

                    # checking if the save file already exists
                    save_file = settings['save_dir'] + '/' + run_name + '.pickle'
                    if num_cpus == 1 and os.path.exists(save_file):
                        print('already exists, not running.')
                    else:
                        command = evolution_slave_script \
                                  + ' --settings "' + str(settings) + '"' \
                                  + ' --field_dict "' + str(field_dict) + '"'
                        s = Slurm(slurm_run_name, slurm_kwargs=slurm_kwargs)
                        s.run(command, local=local)

                    if num_cpus > 1:
                        print('   run set # ' + str(cnt_cpu) + ' / ' + str(num_sets - 1))
                    cnt_cpu += 1

                print('###############')
                cnt_combination += 1
