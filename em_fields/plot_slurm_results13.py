import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns
import copy

# plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set26_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set28_B0_1T_l_10m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set29_B0_1T_l_3m_Post_Rm_2_first_cell_center_crossing/'
# save_dir += '/set30_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
save_dir += '/set31_B0_1T_l_3m_Post_Rm_3_intervals/'

save_dir_curr = save_dir + 'without_RF'
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
E_RF_kVm = 25  # kV/m
# E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

RF_type = 'magnetic_transverse'
B_RF = 0.02  # T
# B_RF = 0.05  # T
# B_RF = 0.1  # T

use_RF = True
# use_RF = False

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'
r_0 = 0
# r_0 = 1.0
# r_0 = 3.0

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
# beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.6, 1.0, 21), 2)  # set28
# alpha_loop_list = alpha_loop_list[10::]
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
# beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

alpha_loop_list = np.round(np.linspace(0.8, 1.0, 11), 2)  # set31
beta_loop_list = np.round(np.linspace(-10, 0, 11), 2)

# vz_over_vth_list = [0.5, 1.0, 1.5]
vz_over_vth_list = [0.5, 0.75, 1.0, 1.25, 1.5]
alpha_const_omega_cyc0_right_list = []
alpha_const_omega_cyc0_left_list = []
for vz_over_vth in vz_over_vth_list:
    alpha_const_omega_cyc0_right_list += [1.0 + beta_loop_list * 2 * np.pi / settings['l']
                                          * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']]
    alpha_const_omega_cyc0_left_list += [1.0 - beta_loop_list * 2 * np.pi / settings['l']
                                         * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']]

# num_t_points_list = [3, 5, 13, 21]
num_t_points_list = [1]  # irrelevant in the saturation method
rate_R = {}
rate_L = {}
selectivity = {}
for num_t_points in num_t_points_list:
    rate_R[num_t_points] = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
    rate_L[num_t_points] = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
    selectivity[num_t_points] = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])


def plot_line_on_heatmap(x_heatmap, y_heatmap, y_line, ax=None, color='b', linewidth=2):
    x_heatmap_normed = 0.5 + np.array(range(len(x_heatmap)))
    y_line_normed = (y_line - y_heatmap[0]) / (y_heatmap[-1] - y_heatmap[0]) * len(y_heatmap) - 0.5
    sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    return


for ind_beta, beta in enumerate(beta_loop_list):
    for ind_alpha, alpha in enumerate(alpha_loop_list):

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
        if absolute_velocity_sampling_type == 'const_vth':
            set_name = 'const_vth_' + set_name
        if r_0 > 0:
            set_name = 'r0_' + str(r_0) + '_' + set_name
        print(set_name)

        save_dir_curr = save_dir + set_name
        # load runs data
        data_dict_file = save_dir_curr + '.pickle'
        with open(data_dict_file, 'rb') as fid:
            data_dict = pickle.load(fid)

        for key in data_dict.keys():
            data_dict[key] = np.array(data_dict[key])

        # divide the phase space by the angle
        if ind_beta == 0 and ind_alpha == 0:
            theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
            N_theta_LC = 1
            N_theta_T = 1
            # N_theta_LC = 2
            # N_theta_T = 2
            # N_theta_LC = 3
            # N_theta_T = 3
            # N_theta_LC = 4
            # N_theta_T = 4
            # N_theta_LC = 2
            # N_theta_T = 5
            # N_theta_LC = 5
            # N_theta_T = 6
            # N_theta_LC = 4
            # N_theta_T = 7
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
                360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)

            # track if a particle left the population, and then cancel counting it for the following times
            if ind_t == 0:
                # particles_counter_mat = np.zeros([N_theta, N_theta])
                inds_bins_ini = []
                cancelled_particles = np.zeros(len(inds_particles))
            particles_counter_mat2 = np.zeros([N_theta, N_theta])

            for ind_p in inds_particles:
                # if not cancelled_particles[ind_p]:
                theta_curr = theta_adjusted[ind_p]
                ind_bin_fin = [k for k, (t1, t2) in enumerate(zip(theta_bins_min_list, theta_bins_max_list))
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

        for ind_num_t_points, num_t_points in enumerate(num_t_points_list):

            nu_decay_list = []
            nu_rise_list = []
            inds_t_array = np.array(range(num_t_points))

            # # for i, ax in enumerate(axs.ravel()):
            # for i in range(N_theta):
            #     # fit to exponential decay
            #     def exp_decay(t, nu):
            #         return np.exp(-nu * t)
            #     p0 = (1.0)
            #     params, cv = scipy.optimize.curve_fit(exp_decay, t_array[inds_t_array],
            #                                           particles_counter_mat2_3d[i, i, inds_t_array], p0)
            #     nu = params[0]
            #     nu_decay_list += [nu]
            #
            # # fit the exponential saturation
            # for i, j in [[0, 1], [N_theta-1, N_theta-2]]:
            #     def exp_saturation(t, nu):
            #         return nu / nu_decay_list[i] * (1.0 - np.exp(-nu_decay_list[i] * t))
            #     p0 = (0.01)
            #     params, cv = scipy.optimize.curve_fit(exp_saturation, t_array[inds_t_array],
            #                                           particles_counter_mat2_3d[i, j, inds_t_array], p0)
            #     nu = params[0]
            #     if i == 0 and j == 1:
            #         nu_rise_list += [nu]
            #     elif i == N_theta - 1 and j == N_theta - 2:
            #         nu_rise_list += [nu]

            # calc the saturattion value
            for i, j in [[0, 1], [N_theta - 1, N_theta - 2]]:
                inds_t_saturation = range(7, 21)
                saturation_value = np.mean(particles_counter_mat2_3d[i, j, inds_t_saturation])
                nu = saturation_value / 1.0
                if i == 0 and j == 1:
                    nu_rise_list += [nu]
                elif i == N_theta - 1 and j == N_theta - 2:
                    nu_rise_list += [nu]

            rate_R[num_t_points][ind_beta, ind_alpha] = nu_rise_list[0]
            rate_L[num_t_points][ind_beta, ind_alpha] = nu_rise_list[1]
            selectivity[num_t_points][ind_beta, ind_alpha] = nu_rise_list[0] / nu_rise_list[1]

### PLOTS

# annot = False
annot = True

# fig, axs = plt.subplots(2, 2, figsize=(16, 8))
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
annot_fontsize = 8
annot_fmt = '.2f'

# for ind_ax, ax in enumerate(axs.ravel()):
ind_ax = 0
ax = axs

num_t_points = num_t_points_list[ind_ax]

ax = axs[0]
y = selectivity[num_t_points]
vmin = np.min(y)
vmax = np.max(y)
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], ax=ax, color='b',
                         linewidth=2)
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], ax=ax, color='b',
                         linewidth=2)
ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
# ax.set_title('selectivity based on fit to ' + str(num_t_points) + ' points')

# ax.set_title('log of ' + ylabel_delta_v)

if use_RF is False:
    title = 'without RF'
elif RF_type == 'electric_transverse':
    title = '$E_{RF}$=' + str(E_RF_kVm) + 'kV/m'
elif RF_type == 'magnetic_transverse':
    title = '$B_{RF}$=' + str(B_RF) + 'T'
# print(title)
title += ', selectivity using the saturation value method'
ax.set_title(title, fontsize=12)

ax = axs[1]
y = rate_R[num_t_points]
vmin = np.min(y)
vmax = np.max(y)
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], ax=ax, color='b',
                         linewidth=2)
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], ax=ax, color='b',
                         linewidth=2)
ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
ax.set_title('$\\nu_R$', fontsize=12)

ax = axs[2]
y = rate_L[num_t_points]
vmin = np.min(y)
vmax = np.max(y)
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], ax=ax, color='b',
                         linewidth=2)
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], ax=ax, color='b',
                         linewidth=2)
ax.set_xlabel('$\\beta$')
ax.set_ylabel('$\\alpha$')
ax.set_title('$\\nu_L$', fontsize=12)

fig.set_tight_layout(0.5)