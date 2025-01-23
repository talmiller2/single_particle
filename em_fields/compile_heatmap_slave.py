#!/usr/bin/env python3

import argparse
import ast
import copy
import os.path
import pickle

import numpy as np
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--passed_dict', help='settings dict for the compilation',
                    type=str, required=True)

args = parser.parse_args()
print('args.passed_dict = ' + str(args.passed_dict))
passed_dict = ast.literal_eval(args.passed_dict)

###########

# extract variables from passed dict
use_RF = passed_dict['use_RF']
alpha_loop_list = passed_dict['alpha_loop_list']
beta_loop_list = passed_dict['beta_loop_list']
save_dir = passed_dict['save_dir']
set_name = passed_dict['set_name']
RF_type = passed_dict['RF_type']
RF_amplitude = passed_dict['RF_amplitude']
induced_fields_factor = passed_dict['induced_fields_factor']
with_kr_correction = passed_dict['with_kr_correction']
time_step_tau_cyclotron_divisions = passed_dict['time_step_tau_cyclotron_divisions']
absolute_velocity_sampling_type = passed_dict['absolute_velocity_sampling_type']
sigma_r0 = passed_dict['sigma_r0']
radial_distribution = passed_dict['radial_distribution']
gas_name = passed_dict['gas_name']
compiled_save_file = passed_dict['compiled_save_file']
Rm = passed_dict['Rm']
l = passed_dict['l']
v_th = passed_dict['v_th']


print('****** compiled_save_file', compiled_save_file)

if os.path.exists(compiled_save_file):
    print('compiled_save_file exists, skipping.')
    pass
else:

    process_names = ['rc', 'lc', 'cr', 'cl', 'rl', 'lr']
    process_index_pairs = [(0, 1), (2, 1), (1, 0), (1, 2), (0, 2), (2, 0)]

    compiled_dict = {}
    compiled_dict['alpha_loop_list'] = alpha_loop_list
    compiled_dict['beta_loop_list'] = beta_loop_list

    num_files = len(alpha_loop_list) * len(beta_loop_list)
    ind_file = 0

    for ind_beta, beta in enumerate(beta_loop_list):
        for ind_alpha, alpha in enumerate(alpha_loop_list):
            ind_file += 1

            set_name = ''
            if use_RF is False:
                set_name += 'without_RF'
            else:
                if RF_type == 'electric_transverse':
                    set_name += 'ERF_' + str(RF_amplitude)
                elif RF_type == 'magnetic_transverse':
                    set_name += 'BRF_' + str(RF_amplitude)
                set_name += '_alpha_' + str(alpha)
                set_name += '_beta_' + str(beta)
                if induced_fields_factor < 1.0:
                    set_name += '_iff' + str(induced_fields_factor)
                if with_kr_correction == True:
                    set_name += '_withkrcor'
            set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
            if absolute_velocity_sampling_type == 'const_vth':
                set_name += '_const_vth'
            if sigma_r0 > 0:
                set_name += '_sigmar' + str(sigma_r0)
                if radial_distribution == 'normal':
                    set_name += 'norm'
                elif radial_distribution == 'uniform':
                    set_name += 'unif'
            set_name += '_' + gas_name

            print('#' + str(ind_file) + '/' + str(num_files) + ': ' + set_name)
            print('loading alpha=' + str(alpha) + ', beta=' + str(beta))

            save_dir_curr = save_dir + set_name
            # load runs data
            data_dict_file = save_dir_curr + '.pickle'
            with open(data_dict_file, 'rb') as fid:
                data_dict = pickle.load(fid)

            # filter out the particles that ended prematurely
            len_t_expected = len(data_dict['t'][0])
            num_particles = len(data_dict['t'])

            if ind_alpha == 0 and ind_beta == 0:

                t_array = np.array(data_dict['t'][0])
                compiled_dict['t_array'] = t_array
                compiled_dict['t_array_normed'] = t_array / (l / v_th)

                zero_mat = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                for process_name in process_names:
                    for suffix in ['end', 'end_std']:
                        compiled_dict['N_' + process_name + '_' + suffix] = copy.deepcopy(zero_mat)

                for key in ['percent_ok', 'E_ratio_mean']:
                    compiled_dict[key] = copy.deepcopy(zero_mat)

                zero_tensor = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list), len(t_array)])
                for process_name in process_names:
                    for suffix in ['curve_mean', 'curve_std']:
                        compiled_dict['N_' + process_name + '_' + suffix] = copy.deepcopy(zero_tensor)

            # calc percent_ok
            inds_ok = []
            for ind_particle, t in enumerate(data_dict['t']):
                if len(t) == len_t_expected:
                    inds_ok += [ind_particle]
            compiled_dict['percent_ok'][ind_beta, ind_alpha] = len(inds_ok) / num_particles * 100

            for key in data_dict.keys():
                data_dict[key] = np.array([data_dict[key][i] for i in inds_ok])

            compiled_dict['E_ratio_mean'][ind_beta, ind_alpha] = np.nanmean(data_dict['v'][:, -1] ** 2) / np.nanmean(
                data_dict['v'][:, 0] ** 2)

            # divide the phase space by the angle
            if ind_beta == 0 and ind_alpha == 0:
                theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(Rm))
                N_theta_LC = 1
                N_theta_T = 1
                N_theta = 2 * N_theta_LC + N_theta_T
                dtheta_LC = theta_LC / N_theta_LC
                dtheta_T = (180 - 2 * theta_LC) / N_theta_T
                theta_bins_max_list = [dtheta_LC]
                for i in range(N_theta_LC - 1):
                    theta_bins_max_list += [theta_bins_max_list[-1] + dtheta_LC]
                for i in range(N_theta_T):
                    theta_bins_max_list += [theta_bins_max_list[-1] + dtheta_T]
                for i in range(N_theta_LC):
                    theta_bins_max_list += [theta_bins_max_list[-1] + dtheta_LC]
                theta_bins_min_list = [0] + theta_bins_max_list[:-1]

            number_of_time_intervals = len(data_dict['t'][0])
            num_bootstrap_samples = 10
            particles_counter_mat_4d = np.zeros(
                [N_theta, N_theta, number_of_time_intervals, num_bootstrap_samples])

            for ind_boot in range(num_bootstrap_samples):
                if ind_boot == 0:
                    inds_particles = range(num_particles)
                else:
                    inds_particles = np.random.randint(low=0, high=num_particles,
                                                       size=num_particles)  # random set of particles

                for ind_t in range(number_of_time_intervals):

                    v = data_dict['v'][inds_particles, ind_t]
                    v0 = data_dict['v'][inds_particles, 0]
                    vt = data_dict['v_transverse'][inds_particles, ind_t]
                    vt0 = data_dict['v_transverse'][inds_particles, 0]
                    vz = data_dict['v_axial'][inds_particles, ind_t]
                    vz0 = data_dict['v_axial'][inds_particles, 0]
                    theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
                    Bz = data_dict['Bz'][inds_particles, ind_t]
                    Bz0 = data_dict['Bz'][inds_particles, 0]
                    vt_adjusted = vt * np.sqrt(
                        Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

                    det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
                    inds_positive = np.where(det > 0)[0]

                    vz_adjusted = np.zeros(len(inds_particles))
                    # vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])
                    vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(
                        det[inds_positive])  # updated criterion

                    theta_adjusted = 90.0 * np.ones(
                        len(inds_particles))  # if det<0 particle probably close to vz=0
                    theta_adjusted[inds_positive] = np.mod(360 / (2 * np.pi) * np.arctan(
                        vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)

                    # initialize
                    if ind_t == 0:
                        inds_bins_ini = [np.nan for _ in range(num_particles)]

                    particles_counter_mat = np.zeros([N_theta, N_theta])

                    for ind_p in inds_particles:
                        theta_curr = theta_adjusted[ind_p]
                        ind_bin_fin = [k for k, (t1, t2) in
                                       enumerate(zip(theta_bins_min_list, theta_bins_max_list))
                                       if theta_curr > t1 and theta_curr <= t2][0]
                        if ind_t == 0:
                            inds_bins_ini[ind_p] = ind_bin_fin
                        ind_bin_ini = inds_bins_ini[ind_p]

                        particles_counter_mat[ind_bin_ini, ind_bin_fin] += 1

                    if ind_t == 0:
                        N0 = copy.deepcopy(np.diag(particles_counter_mat))

                    particles_counter_mat_4d[:, :, ind_t, ind_boot] = particles_counter_mat

                # divide all densities by the parent initial density
                for ind_t in range(number_of_time_intervals):
                    for ind_bin in range(N_theta):
                        particles_counter_mat_4d[ind_bin, :, ind_t, ind_boot] /= (1.0 * N0[ind_bin])

            # compile the results from the bootstrap procedure
            for process_name, process_index_pair in zip(process_names, process_index_pairs):
                pi, pj = process_index_pair[0], process_index_pair[1]
                compiled_dict['N_' + process_name + '_curve_mean'][ind_beta, ind_alpha] = np.mean(
                    particles_counter_mat_4d[pi, pj, :, :], axis=1)
                compiled_dict['N_' + process_name + '_curve_std'][ind_beta, ind_alpha] = np.std(
                    particles_counter_mat_4d[pi, pj, :, :], axis=1)

                num_t_inds_avg = 5
                inds_t_avg = range(number_of_time_intervals - num_t_inds_avg,
                                   number_of_time_intervals)  # only the last few indices
                compiled_dict['N_' + process_name + '_end'][ind_beta, ind_alpha] = np.mean(
                    particles_counter_mat_4d[pi, pj, inds_t_avg, 0])
                compiled_dict['N_' + process_name + '_end_std'][ind_beta, ind_alpha] = np.std(
                    particles_counter_mat_4d[pi, pj, inds_t_avg, :])

    # save the compiled data for all alpha, beta matrix
    savemat(compiled_save_file, compiled_dict)
