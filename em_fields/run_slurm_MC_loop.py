import os
import pickle

from scipy.stats import maxwell
from slurmpy.slurmpy import Slurm

from em_fields.slurm_functions import get_script_evolution_slave_fenchel2

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel2()

import matplotlib.pyplot as plt

from em_fields.default_settings import define_default_settings, define_default_field

import numpy as np
from scipy.io import savemat

slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

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
save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'

plt.close('all')

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 15), 2) # set24
# lambda_RF_loop_list = np.round(np.linspace(-20, 20, 10), 0)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set25
# lambda_RF_loop_list = np.round(np.linspace(-6, 6, 10), 0)
# lambda_RF_loop_list += np.sign(lambda_RF_loop_list)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
beta_loop_list = np.round(np.linspace(-1, 1, 11), 21)

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
E_RF_kVm = 10  # kV/m
# E_RF_kVm = 30  # kV/m
# E_RF_kVm = 100  # kV/m

# RF_type = 'magnetic_transverse'
B_RF = 0.05  # T

use_RF = True
# use_RF = False
if use_RF is False:
    E_RF_kVm = 0
    alpha_loop_list = [1]
    beta_loop_list = [0]

totol_loop_runs = len(beta_loop_list) * len(alpha_loop_list)
print('totol_loop_runs = ' + str(totol_loop_runs))

# divide the points to a given number of cpus (250 is max in partition core)
num_cpus = 1
# num_cpus = 10
# num_cpus = 50
# num_cpus = 200

cnt_loop = 1

for beta_loop in beta_loop_list:
    for alpha_loop in alpha_loop_list:
        print('loop run ' + str(cnt_loop) + '/' + str(totol_loop_runs)
              + ': alpha=' + str(alpha_loop) + ', beta=' + str(beta_loop))

        # define settings
        settings = {}
        # settings['trajectory_save_method'] = 'intervals'
        settings['stop_criterion'] = 'first_cell_center_crossing'

        # settings['l'] = 1.0  # m (MM cell size)
        settings['l'] = 3.0  # m (MM cell size)

        # settings['absolute_velocity_sampling_type'] = 'const_vth'
        settings['absolute_velocity_sampling_type'] = 'maxwell'

        # settings['direction_velocity_sampling_type'] = 'deterministic'

        # settings['r_0'] = 3.0

        settings = define_default_settings(settings)

        field_dict = {}

        # field_dict['Rm'] = 1.3  # mirror ratio
        # field_dict['Rm'] = 2.0  # mirror ratio
        field_dict['Rm'] = 3.0  # mirror ratio
        # field_dict['Rm'] = 4.0  # mirror ratio
        # field_dict['Rm'] = 6.0  # mirror ratio

        if RF_type == 'electric_transverse':
            field_dict['RF_type'] = 'electric_transverse'
            field_dict['E_RF_kVm'] = E_RF_kVm
        elif RF_type == 'magnetic_transverse':
            field_dict['RF_type'] = 'magnetic_transverse'
            field_dict['B_RF'] = B_RF

        # field_dict['phase_RF_addition'] = 0
        # field_dict['phase_RF_addition'] = np.pi

        field_dict['alpha_RF_list'] = [alpha_loop]
        field_dict['beta_RF_list'] = [beta_loop]

        field_dict['mirror_field_type'] = 'post'
        # field_dict['mirror_field_type'] = 'logan'

        field_dict = define_default_field(settings, field_dict)

        # simulation duration
        # settings['num_snapshots'] = 30
        settings['num_snapshots'] = 50
        # settings['num_snapshots'] = 200
        # settings['num_snapshots'] = 300

        # tmax_mirror_lengths = 1
        tmax_mirror_lengths = 3
        # tmax_mirror_lengths = 5
        # tmax_mirror_lengths = 100
        # tmax_mirror_lengths = 300
        sim_cyclotron_periods = int(
            tmax_mirror_lengths * settings['l'] / settings['v_th'] / field_dict['tau_cyclotron'])
        settings['sim_cyclotron_periods'] = sim_cyclotron_periods

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
        if settings['absolute_velocity_sampling_type'] == 'const_vth':
            run_name = 'const_vth_' + run_name
        if settings['r_0'] > 0:
            run_name = 'r0_' + str(settings['r_0']) + '_' + run_name
        print('run_name: ' + str(run_name))
        settings['run_name'] = run_name

        if num_cpus == 1:
            settings['save_dir'] = save_dir
        else:
            settings['save_dir'] = save_dir + '/' + run_name

        # total_number_of_points = 1
        # total_number_of_points = 40
        # total_number_of_points = 400
        total_number_of_points = 1000
        # total_number_of_points = 2000
        # total_number_of_points = 5000
        # total_number_of_points = 10000
        # total_number_of_points = 20000

        # allow reproducibility
        np.random.seed(0)

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
        points_dict = {'v_0': v_0}

        # random RF phases for each particle
        if settings['apply_random_RF_phase']:
            points_dict['phase_RF'] = 2 * np.pi * np.random.rand(total_number_of_points)

        # save the run settings
        if cnt_loop == 1:
            os.makedirs(settings['save_dir'], exist_ok=True)
            os.chdir(settings['save_dir'])

            settings_file = settings['save_dir'] + '/settings.pickle'
            with open(settings_file, 'wb') as handle:
                pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
            field_dict_file = settings['save_dir'] + '/field_dict.pickle'
            with open(field_dict_file, 'wb') as handle:
                pickle.dump(field_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            points_dict_file = settings['save_dir'] + '/points_dict.mat'
            savemat(points_dict_file, points_dict)

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

        # run the slave_fenchel scripts on multiple cpus
        cnt = 0
        for ind_set, points_set in enumerate(points_set_list):
            settings['ind_set'] = ind_set
            settings['points_set'] = points_set

            print('###############')
            if num_cpus == 1:
                slurm_run_name = run_name
                settings['ind_set'] = None  # TODO ?
            elif num_cpus > 1:
                slurm_run_name = 'set_' + str(ind_set) + '_' + run_name
            print('run_name = ' + run_name)

            command = evolution_slave_fenchel_script \
                      + ' --settings "' + str(settings) + '"' \
                      + ' --field_dict "' + str(field_dict) + '"'
            s = Slurm(slurm_run_name, slurm_kwargs=slurm_kwargs)
            s.run(command)

            if num_cpus > 1:
                print('   run set # ' + str(cnt) + ' / ' + str(num_sets - 1))
            cnt += 1

        cnt_loop += 1
