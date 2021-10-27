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

main_folder = '/home/talm/code/single_particle/slurm_runs/'
# main_folder += '/set5/'
# main_folder += '/set6/'
# main_folder += '/set7_T_10keV_B0_1T_Rm_2_l_1m/'
main_folder += '/set8_T_10keV_B0_1T_Rm_2_l_1m/'

plt.close('all')

# v_loop_list = np.round(np.linspace(0.9, 2.5, 10), 2)
# alpha_loop_list = np.round(np.linspace(0.5, 2, 10), 2)

v_loop_list = [1]
alpha_loop_list = [1]

# v_loop_list = [0.5, 1.0, 1.5, 2.0]
# alpha_loop_list = [1.0, 1.2, 1.5, 2.0]

# v_loop_list = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
# alpha_loop_list = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

totol_loop_runs = len(v_loop_list) * len(alpha_loop_list)
print('totol_loop_runs = ' + str(totol_loop_runs))

cnt_loop = 1

for v_loop in v_loop_list:
    for alpha_loop in alpha_loop_list:
        print('loop run ' + str(cnt_loop) + '/' + str(totol_loop_runs)
              + ': v=' + str(v_loop) + ', alpha=' + str(alpha_loop))
        cnt_loop += 1

        # define settings
        settings = {}
        settings = define_default_settings()

        field_dict = {}

        # field_dict['E_RF_kVm'] = 0  # kV/m
        # field_dict['E_RF_kVm'] = 1  # kV/m
        # field_dict['E_RF_kVm'] = 2  # kV/m
        # field_dict['E_RF_kVm'] = 5  # kV/m
        field_dict['E_RF_kVm'] = 10  # kV/m

        field_dict['v_z_factor_list'] = [v_loop]

        field_dict['alpha_detune_list'] = [alpha_loop for i in range(len(field_dict['v_z_factor_list']))]

        # field_dict['nullify_RF_magnetic_field'] = True

        field_dict = define_default_field(settings, field_dict=field_dict)

        # simulation duration
        sim_cyclotron_periods = int(20 * settings['l'] / settings['v_th'] / field_dict['tau_cyclotron'])
        settings['sim_cyclotron_periods'] = sim_cyclotron_periods

        save_dir = ''
        # save_dir += 'tmax_' + str(settings['sim_cyclotron_periods'])
        # save_dir += '_B0_' + str(field_dict['B0'])
        # save_dir += '_T_' + str(settings['T_keV'])

        if field_dict['E_RF_kVm'] > 0:
            save_dir += 'ERF_' + str(field_dict['E_RF_kVm'])
            save_dir += '_alpha_' + '_'.join([str(alpha_detune) for alpha_detune in field_dict['alpha_detune_list']])
            save_dir += '_vz_' + '_'.join([str(v_z_factor) for v_z_factor in field_dict['v_z_factor_list']])
        else:
            save_dir = 'ERF_0'

        if field_dict['nullify_RF_magnetic_field']:
            save_dir += '_zeroBRF'

        print('save_dir: ' + str(save_dir))

        settings['save_dir'] = main_folder + '/' + save_dir
        os.makedirs(settings['save_dir'], exist_ok=True)
        os.chdir(settings['save_dir'])

        settings_file = settings['save_dir'] + '/settings.pickle'
        with open(settings_file, 'wb') as handle:
            pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        field_dict_file = settings['save_dir'] + '/field_dict.pickle'
        with open(field_dict_file, 'wb') as handle:
            pickle.dump(field_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # total_number_of_points = 40
        # total_number_of_points = 1000
        total_number_of_points = 10000
        # total_number_of_points = 20000

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
        else:
            raise ValueError('invalid direction_velocity_sampling_type :'
                             + str(settings['direction_velocity_sampling_type']))

        # total velocity vector
        v_0 = rand_unit_vec
        for i in range(total_number_of_points):
            v_0[i, :] *= v_abs_samples[i]

        # create and save the points file to be run later
        points_dict = {'v_0': v_0}
        points_dict_file = settings['save_dir'] + '/points_dict.mat'
        savemat(points_dict_file, points_dict)

        # divide the points to a given number of cpus (250 is max in partition core)
        # num_cpus = 2
        num_cpus = 15
        # num_cpus = 50
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
            run_name = 'set_' + str(ind_set) + '_' + save_dir
            print('run_name = ' + run_name)

            settings['ind_set'] = ind_set
            settings['points_set'] = points_set

            command = evolution_slave_fenchel_script \
                      + ' --settings "' + str(settings) + '"' \
                      + ' --field_dict "' + str(field_dict) + '"'
            s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
            s.run(command)
            print('run set # ' + str(cnt) + ' / ' + str(num_sets - 1))
            cnt += 1
