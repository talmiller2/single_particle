import copy
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
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
# save_dir += '/set45_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
save_dir += '/set46_B0_2T_l_1m_Post_Rm_3_intervals_D_T/'

save_dir_curr = save_dir + 'without_RF'
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

# RF_type = 'electric_transverse'
# E_RF_kVm = 0.1 # kV/m
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
E_RF_kVm = 25  # kV/m
# E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

RF_type = 'magnetic_transverse'
# B_RF = 0.001  # T
# B_RF = 0.01  # T
# B_RF = 0.02  # T
# B_RF = 0.04  # T
# B_RF = 0.08  # T

use_RF = True
# use_RF = False

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'


# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
# beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.6, 1.0, 21), 2)  # set28
# alpha_loop_list = alpha_loop_list[10::]
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
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

alpha_loop_list = np.round(np.linspace(0.7, 1.3, 11), 2)  # set43
beta_loop_list = np.round(np.linspace(-2, 2, 11), 2)

gas_name_list = ['deuterium', 'tritium']
# sigma_r0_list = [0, 0.1]
# induced_fields_factor_list = [1, 0.5, 0]
B_RF_list = [0.02, 0.04, 0.08]  # [T]
sigma_r0_list = [0.1]
induced_fields_factor_list = [1, 0]
with_RF_xy_corrections = True
time_step_tau_cyclotron_divisions = 40

for B_RF in B_RF_list:
    for gas_name in gas_name_list:
        for sigma_r0 in sigma_r0_list:
            for induced_fields_factor in induced_fields_factor_list:

                ## save compiled data to file
                set_name = 'compiled_'
                if RF_type == 'electric_transverse':
                    set_name += 'ERF_' + str(E_RF_kVm)
                elif RF_type == 'magnetic_transverse':
                    set_name += 'BRF_' + str(B_RF)
                if induced_fields_factor < 1.0:
                    set_name += '_iff' + str(induced_fields_factor)
                if with_RF_xy_corrections == False:
                    set_name += '_woxyRFcor'
                set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
                if absolute_velocity_sampling_type == 'const_vth':
                    set_name += '_const_vth'
                if sigma_r0 > 0:
                    set_name += '_sigmar' + str(sigma_r0)
                set_name += '_' + gas_name
                save_file = save_dir + '/' + set_name + '.mat'
                print('****** save_file', save_file)

                if os.path.exists(save_file):
                    pass
                else:

                    num_files = len(alpha_loop_list) * len(beta_loop_list)
                    ind_file = 0

                    N_rc = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                    N_lc = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                    N_cr = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                    N_cl = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                    selectivity = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                    percent_ok = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                    dE_mean = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
                    dE_std = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])

                    for ind_beta, beta in enumerate(beta_loop_list):
                        for ind_alpha, alpha in enumerate(alpha_loop_list):
                            ind_file += 1

                            # print('loading alpha=' + str(alpha) + ', beta=' + str(beta))
                            # try:

                            set_name = ''
                            if use_RF is False:
                                set_name += 'without_RF'
                            else:
                                if RF_type == 'electric_transverse':
                                    set_name += 'ERF_' + str(E_RF_kVm)
                                elif RF_type == 'magnetic_transverse':
                                    set_name += 'BRF_' + str(B_RF)
                                set_name += '_alpha_' + str(alpha)
                                set_name += '_beta_' + str(beta)
                                if induced_fields_factor < 1.0:
                                    set_name += '_iff' + str(induced_fields_factor)
                                if with_RF_xy_corrections == False:
                                    set_name += '_woxyRFcor'
                            set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
                            if absolute_velocity_sampling_type == 'const_vth':
                                set_name += '_const_vth'
                            if sigma_r0 > 0:
                                set_name += '_sigmar' + str(sigma_r0)
                            set_name += '_' + gas_name

                            # print(set_name)
                            print('#' + str(ind_file) + '/' + str(num_files) + ': ' + set_name)

                            save_dir_curr = save_dir + set_name
                            # load runs data
                            data_dict_file = save_dir_curr + '.pickle'
                            with open(data_dict_file, 'rb') as fid:
                                data_dict = pickle.load(fid)

                            # for key in data_dict.keys():
                            #     data_dict[key] = np.array(data_dict[key])

                            # filter out the particles that ended prematurely
                            num_particles = len(data_dict['t'])
                            inds_ok = []
                            for ind_particle, t in enumerate(data_dict['t']):
                                if len(t) == 30:
                                    inds_ok += [ind_particle]
                            percent_ok_curr = len(inds_ok) / num_particles * 100
                            for key in data_dict.keys():
                                data_dict[key] = np.array([data_dict[key][i][0:30] for i in inds_ok])

                            # divide the phase space by the angle
                            if ind_beta == 0 and ind_alpha == 0:
                                theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
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

                            # number_of_time_intervals = 3
                            number_of_time_intervals = data_dict['t'].shape[1]
                            # number_of_time_intervals = 5

                            # particles_counter_mat_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])
                            particles_counter_mat2_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])

                            for ind_t in range(number_of_time_intervals):

                                inds_particles = range(data_dict['t'].shape[0])
                                # inds_particles = [0, 1, 2]
                                # inds_particles = range(1001)
                                # if ind_t == 0:
                                #     print('num particles = ' + str(len(inds_particles)))

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
                                vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])
                                # vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(det[inds_positive])

                                theta_adjusted = 90.0 * np.ones(len(inds_particles))
                                theta_adjusted[inds_positive] = np.mod(
                                    360 / (2 * np.pi) * np.arctan(
                                        vt_adjusted[inds_positive] / vz_adjusted[inds_positive]),
                                    180)

                                # track if a particle left the population, and then cancel counting it for the following times
                                if ind_t == 0:
                                    # particles_counter_mat = np.zeros([N_theta, N_theta])
                                    inds_bins_ini = []
                                    cancelled_particles = np.zeros(len(inds_particles))
                                particles_counter_mat2 = np.zeros([N_theta, N_theta])

                                for ind_p in inds_particles:
                                    # if not cancelled_particles[ind_p]:
                                    theta_curr = theta_adjusted[ind_p]
                                    ind_bin_fin = \
                                        [k for k, (t1, t2) in enumerate(zip(theta_bins_min_list, theta_bins_max_list))
                                         if theta_curr > t1 and theta_curr <= t2][0]
                                    if ind_t == 0:
                                        inds_bins_ini += [ind_bin_fin]
                                        # particles_counter_mat[ind_bin_fin, ind_bin_fin] += 1

                                    ind_bin_ini = inds_bins_ini[ind_p]

                                    particles_counter_mat2[ind_bin_ini, ind_bin_fin] += 1
                                    # if ind_bin_fin != ind_bin_ini:
                                    #     if not cancelled_particles[ind_p]:
                                    #         particles_counter_mat[ind_bin_ini, ind_bin_ini] -= 1
                                    #         particles_counter_mat[ind_bin_ini, ind_bin_fin] += 1
                                    #         cancelled_particles[ind_p] = 1

                                if ind_t == 0:
                                    N0 = copy.deepcopy(np.diag(particles_counter_mat2))

                                # particles_counter_mat_3d[:, :, ind_t] = particles_counter_mat
                                particles_counter_mat2_3d[:, :, ind_t] = particles_counter_mat2

                            # divide all densities by the parent initial density
                            for ind_t in range(number_of_time_intervals):
                                for ind_bin in range(N_theta):
                                    # particles_counter_mat_3d[ind_bin, :, ind_t] /= (1.0 * N0[ind_bin])
                                    particles_counter_mat2_3d[ind_bin, :, ind_t] /= (1.0 * N0[ind_bin])

                            t_array = data_dict['t'][0]
                            t_array /= settings['l'] / settings['v_th']

                            nu_rise_list = []
                            # calc the saturattion value
                            for i, j in [[0, 1], [N_theta - 1, N_theta - 2]]:
                                # inds_t_saturation = range(7, 21)
                                # inds_t_saturation = range(15, 31) # for 2023 paper
                                inds_t_saturation = range(15, 30)
                                # inds_t_saturation = range(len(t_array))
                                saturation_value = np.mean(particles_counter_mat2_3d[i, j, inds_t_saturation])
                                nu = saturation_value / 1.0
                                if i == 0 and j == 1:
                                    nu_rise_list += [nu]
                                elif i == N_theta - 1 and j == N_theta - 2:
                                    nu_rise_list += [nu]

                            N_rc[ind_beta, ind_alpha] = np.mean(particles_counter_mat2_3d[0, 1, inds_t_saturation])
                            N_lc[ind_beta, ind_alpha] = np.mean(particles_counter_mat2_3d[2, 1, inds_t_saturation])
                            N_cr[ind_beta, ind_alpha] = np.mean(particles_counter_mat2_3d[1, 0, inds_t_saturation])
                            N_cl[ind_beta, ind_alpha] = np.mean(particles_counter_mat2_3d[1, 2, inds_t_saturation])
                            selectivity[ind_beta, ind_alpha] = N_rc[ind_beta, ind_alpha] / N_lc[ind_beta, ind_alpha]
                            percent_ok[ind_beta, ind_alpha] = percent_ok_curr
                            dE_mean[ind_beta, ind_alpha] = np.mean((data_dict['v'][inds_particles, -1] / data_dict['v'][
                                inds_particles, 0]) ** 2)
                            dE_std[ind_beta, ind_alpha] = np.std((data_dict['v'][inds_particles, -1] / data_dict['v'][
                                inds_particles, 0]) ** 2)
                            # except:
                            #     print('*** FAILED ***')

                    mat_dict = {}
                    mat_dict['alpha_loop_list'] = alpha_loop_list
                    mat_dict['beta_loop_list'] = beta_loop_list
                    mat_dict['N_rc'] = N_rc
                    mat_dict['N_lc'] = N_lc
                    mat_dict['N_cr'] = N_cr
                    mat_dict['N_cl'] = N_cl
                    mat_dict['selectivity'] = selectivity
                    mat_dict['percent_ok'] = percent_ok
                    mat_dict['dE_mean'] = dE_mean
                    mat_dict['dE_std'] = dE_std

                    savemat(save_file, mat_dict)
