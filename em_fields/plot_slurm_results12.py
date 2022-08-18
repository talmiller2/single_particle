import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np
import copy

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

# figsize_large = (16, 9)
figsize_large = (14, 7)

# plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set26_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set28_B0_1T_l_10m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set29_B0_1T_l_3m_Post_Rm_2_first_cell_center_crossing/'
# save_dir += '/set30_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set31_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set32_B0_1T_l_1m_Post_Rm_3_intervals/'
save_dir += '/set33_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set34_B0_1T_l_3m_Post_Rm_3_intervals/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 25  # kV/m
# E_RF_kVm = 50  # kV/m
E_RF_kVm = 100  # kV/m

# RF_type = 'magnetic_transverse'
# B_RF = 0.01  # T
# B_RF = 0.02  # T
B_RF = 0.04  # T
# B_RF = 0.05  # T
# B_RF = 0.1  # T

use_RF = True
# use_RF = False

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'
r_0 = 0
# r_0 = 1.5
# r_0 = 3.0

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
# beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.6, 1.0, 21), 2)  # set28
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
# beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 11), 2)  # set31, 32, 33
# beta_loop_list = np.round(np.linspace(-10, 0, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set34
# beta_loop_list = np.round(np.linspace(-5, 5, 11), 2)

# for ind_beta, beta_RF in enumerate(beta_loop_list):
#     for ind_alpha, alpha_RF in enumerate(alpha_loop_list):
# ind_alpha = 0
# ind_alpha = 1
# ind_alpha = 2
# ind_alpha = 4
# ind_alpha = 5
# ind_alpha = 7
# ind_beta = 1
# ind_beta = 2
# ind_beta = 5
# ind_beta = 4
# ind_beta = 3
# alpha = alpha_loop_list[ind_alpha]
# beta = beta_loop_list[ind_beta]
#
alpha = 0.8
# alpha = 0.82
# alpha = 0.85
# alpha = 0.86
# alpha = 0.9
# alpha = 0.92
# alpha = 0.94
# alpha = 0.95
# alpha = 0.96
# alpha = 0.98
# alpha = 0.99
# alpha = 1.0
# alpha = 1.01
# alpha = 1.02
# alpha = 1.04
# alpha = 1.05
# alpha = 1.1

# beta = 0.0
# beta = -0.1
# beta = -0.2
# beta = -0.3
# beta = -0.4
# beta = -0.5
# beta = -0.7
# beta = -1.0
# beta = -2.0
# beta = -2.5
# beta = -3.0
# beta = -3.75
# beta = -4.0
# beta = -4.5
# beta = -5.0
# beta = -6.0
# beta = -7.5
# beta = -8.0
# beta = -9.0
beta = -10.0
# beta = 1.0


if use_RF is False:
    title = 'without RF'
elif RF_type == 'electric_transverse':
    title = '$E_{RF}$=' + str(E_RF_kVm) + 'kV/m, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
elif RF_type == 'magnetic_transverse':
    title = '$B_{RF}$=' + str(B_RF) + 'T, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
print(title)

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
    set_name += '_r0_' + str(r_0) + '_' + set_name
# set_name += '_antiresonant'

save_dir_curr = save_dir + set_name

# load runs data
data_dict_file = save_dir_curr + '.pickle'
with open(data_dict_file, 'rb') as fid:
    data_dict = pickle.load(fid)
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

for key in data_dict.keys():
    data_dict[key] = np.array(data_dict[key])

# divide the phase space by the angle

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

# for k, (t1, t2) in enumerate(zip(theta_bins_min_list, theta_bins_max_list)):
#     print(k, (t1, t2))

num_rows = 1
num_cols = 3
# num_rows = 3
# num_cols = 3
# num_rows = 4
# num_cols = 4

# number_of_time_intervals = 3
number_of_time_intervals = data_dict['t'].shape[1]

# densities_dict = {}
# for i in range(N_theta):
#     densities_dict[i] = []

particles_counter_mat_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])
particles_counter_mat2_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])

from matplotlib import cm

colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
# colors = ['b', 'g', 'r']
# colors = ['r', 'g', 'b']

for ind_t in range(number_of_time_intervals):
    # for ind_t in [0, 10]:
    # for ind_t in [0, 1]:
    # for ind_t in [0, 10, 20]:
    #     print(ind_t)

    inds_particles = range(data_dict['t'].shape[0])
    # inds_particles = [0, 1, 2]
    # inds_particles = range(1001)
    if ind_t == 0:
        print('num particles = ' + str(len(inds_particles)))

    v = data_dict['v'][inds_particles, ind_t]
    v0 = data_dict['v'][inds_particles, 0]
    vt = data_dict['v_transverse'][inds_particles, ind_t]
    vt0 = data_dict['v_transverse'][inds_particles, 0]
    vz = data_dict['v_axial'][inds_particles, ind_t]
    vz0 = data_dict['v_axial'][inds_particles, 0]
    theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
    Bz = data_dict['Bz'][inds_particles, ind_t]
    Bz0 = data_dict['Bz'][inds_particles, 0]
    vt_adjusted = vt * np.sqrt(Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

    # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt0 ** 2.0 * (Bz / Bz0 - 1))
    # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz))
    # theta_adjusted = np.mod(360 / (2 * np.pi) * np.arctan(vt_adjusted / vz_adjusted), 180)

    det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
    inds_positive = np.where(det > 0)[0]
    vz_adjusted = np.zeros(len(inds_particles))
    vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])
    # vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(det[inds_positive])

    theta_adjusted = 90.0 * np.ones(len(inds_particles))
    theta_adjusted[inds_positive] = np.mod(
        360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)

    # v_adjusted = np.sqrt(vt_adjusted ** 2.0 + vz_adjusted ** 2.0)

    # print('mean of vt_adjusted / vt0 = ' + str(np.mean(vt_adjusted) / np.mean(vt0)))
    # print('mean of vz_adjusted / vz0 = ' + str(np.mean(vz_adjusted) / np.mean(vz0)))
    # print('mean of v / vadj = ' + str(np.mean(v_adjusted) / np.mean(v)))
    # print('mean of v / v0 = ' + str(np.mean(v) / np.mean(v0)))

    # print('mean of vt_adjusted / vt0 = ' + str(np.mean(vt_adjusted / vt0)))
    # print('mean of vz_adjusted / vz0 = ' + str(np.mean(vz_adjusted / vz0)))
    # print('mean of v / vadj = ' + str(np.mean(v_adjusted / v)))
    # print('mean of v / v0 = ' + str(np.mean(v / v0)))

    color = colors[ind_t]

    # if ind_t == 0:
    #     fig2, ax1 = plt.subplots(1, 1, figsize=(7, 7), num=2)
    # if np.mod(ind_t, 5) == 1:
    #     label = str(ind_t)
    #     label = '$t \\cdot v_{th} / l$=' + '{:.2f}'.format(
    #         data_dict['t'][0, ind_t] / (settings['l'] / settings['v_th']))
    #     ax1.scatter(vz_adjusted / settings['v_th'], vt_adjusted / settings['v_th'], color=color, alpha=0.2, label=label)
    #     if ind_t == 0:
    #         # plot the diagonal LC lines
    #         vz_axis = np.array([0, 2 * settings['v_th']])
    #         vt_axis = vz_axis * np.sqrt(1 / (field_dict['Rm'] - 1))
    #         ax1.plot(vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
    #         ax1.plot(-vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
    #     ax1.set_xlabel('$v_z / v_{th}$')
    #     ax1.set_ylabel('$v_{\\perp} / v_{th}$')
    #     ax1.legend()
    #     ax1.grid(True)

    # track if a particle left the population, and then cancel counting it for the following times
    if ind_t == 0:
        particles_counter_mat = np.zeros([N_theta, N_theta])
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
            particles_counter_mat[ind_bin_fin, ind_bin_fin] += 1

        ind_bin_ini = inds_bins_ini[ind_p]

        particles_counter_mat2[ind_bin_ini, ind_bin_fin] += 1
        if ind_bin_fin != ind_bin_ini:
            if not cancelled_particles[ind_p]:
                particles_counter_mat[ind_bin_ini, ind_bin_ini] -= 1
                particles_counter_mat[ind_bin_ini, ind_bin_fin] += 1
                cancelled_particles[ind_p] = 1

    if ind_t == 0:
        N0 = copy.deepcopy(np.diag(particles_counter_mat))
        # print(N0)

    # divide all densities by the parent initial density
    # for i in range(N_theta):
    #     particles_counter_mat[i, :] /= (1.0 * N0[i])
    # print(particles_counter_mat)

    particles_counter_mat_3d[:, :, ind_t] = particles_counter_mat
    particles_counter_mat2_3d[:, :, ind_t] = particles_counter_mat2

    ## plot the evolution of particles in a specific bin
    # if ind_t == 0:
    #     fig3, axs3 = plt.subplots(num_rows, num_cols, figsize=figsize_large, num=3)
    #     fig3.suptitle(title)
    #
    # for ind_bin in range(N_theta):
    #     ax = axs3.ravel()[ind_bin]
    #
    #     inds_p_curr_bin = np.where(np.array(inds_bins_ini) == ind_bin)[0]
    #     if ind_t > 0:
    #         inds_p_curr_bin_not_cancelled = []
    #         inds_p_curr_bin_cancelled = []
    #         for ind_p in inds_p_curr_bin:
    #             if cancelled_particles[ind_p] == 1:
    #                 inds_p_curr_bin_cancelled += [ind_p]
    #             else:
    #                 inds_p_curr_bin_not_cancelled += [ind_p]
    #
    #         # plot the diagonal LC lines
    #         vz_axis = np.array([0, 2 * settings['v_th']])
    #         vt_axis = vz_axis * np.sqrt(1 / (field_dict['Rm'] - 1))
    #         ax.plot(vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='grey', linestyle='-')
    #         ax.plot(-vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='grey', linestyle='-')
    #         theta_min = theta_bins_min_list[ind_bin]
    #         theta_max = theta_bins_max_list[ind_bin]
    #         ax.plot(np.cos(2 * np.pi / 360 * theta_min) * vz_axis / settings['v_th'], np.sin(2 * np.pi / 360 * theta_min) * vz_axis / settings['v_th'] , color='k', linestyle=':')
    #         ax.plot(np.cos(2 * np.pi / 360 * theta_max) * vz_axis / settings['v_th'], np.sin(2 * np.pi / 360 * theta_max) * vz_axis / settings['v_th'], color='k', linestyle=':')
    #     if ind_t == 0:
    #         color = 'b'
    #     elif ind_t == len(data_dict['t'][0]) - 1:
    #         color = 'r'
    #     # if ind_t in [0, len(data_dict['t'][0]) - 1]:
    #     if ind_t in [len(data_dict['t'][0]) - 1]:
    #         # ax5.scatter(vz_adjusted[inds_p_curr_bin] / settings['v_th'], vt_adjusted[inds_p_curr_bin] / settings['v_th'], color=color, alpha=0.2)
    #         # ax5.scatter(vz_adjusted[inds_p_curr_bin_cancelled] / settings['v_th'], vt_adjusted[inds_p_curr_bin_cancelled] / settings['v_th'], color='r', alpha=0.2)
    #         # ax5.scatter(vz_adjusted[inds_p_curr_bin_not_cancelled] / settings['v_th'], vt_adjusted[inds_p_curr_bin_not_cancelled] / settings['v_th'], color='b', alpha=0.2)
    #         ax.scatter(vz0[inds_p_curr_bin_cancelled] / settings['v_th'], vt0[inds_p_curr_bin_cancelled] / settings['v_th'], color='r', alpha=0.2)
    #         ax.scatter(vz0[inds_p_curr_bin_not_cancelled] / settings['v_th'], vt0[inds_p_curr_bin_not_cancelled] / settings['v_th'], color='b', alpha=0.2)
    #         # ax.set_xlabel('$v_z / v_{th}$')
    #         # ax.set_ylabel('$v_{\\perp} / v_{th}$')
    #         # ax.set_title('following particles in bin #' + str(ind_bin + 1))
    #         ax.set_title('bin #' + str(ind_bin + 1))
    #         # ax.legend()
    #         ax.grid(True)
    #         ax.set_xlim([-2, 2])
    #         ax.set_ylim([0, 2])
    #         # fig5.set_tight_layout(0.5)
    #         fig3.set_tight_layout({'pad': 0.5, 'rect': (0, 0, 1, 0.95)})

# divide all densities by the parent initial density
for ind_t in range(number_of_time_intervals):
    for ind_bin in range(N_theta):
        particles_counter_mat_3d[ind_bin, :, ind_t] /= (1.0 * N0[ind_bin])
        particles_counter_mat2_3d[ind_bin, :, ind_t] /= (1.0 * N0[ind_bin])
particles_counter_mat2_for_fit_3d = copy.deepcopy(particles_counter_mat2_3d)

t_array = data_dict['t'][0]
t_array /= settings['l'] / settings['v_th']

# for i in range(N_theta):
#     densities_dict[i] = np.array(densities_dict[i])
#     densities_dict[i] = densities_dict[i] / (1.0 * densities_dict[i][0])


colors = cm.rainbow(np.linspace(0, 1, N_theta))
nu_decay_list = []
nu_mat = np.zeros([N_theta, N_theta])

do_fit = True
# do_fit = False

# inds_t_array = np.array(range(8)
# inds_t_array = np.array(range(13))
# inds_t_array = np.array([0, 1, 2, 3, 4])
# inds_t_array = np.array([0, 1, 2])
inds_t_array = range(len(t_array))
fig3, axs3 = plt.subplots(num_rows, num_cols, figsize=figsize_large, num=1)
fig3.suptitle(title)

# for i, ax in enumerate(axs.ravel()):
for i in range(N_theta):
    ax = axs3.ravel()[i]
    for j in range(N_theta):
        if j in [i - 1, i, i + 1]:
            # if j in [i - 1, i + 1]:
            if j == i:
                color = 'g'
            elif j == i - 1:
                color = 'r'
            elif j == i + 1:
                color = 'b'

            # color = colors[j]
            # linestyle = '-'
            # # linestyle = '--'
            # ax.plot(t_array, particles_counter_mat_3d[i, j, :], color=color, linestyle=linestyle)
            linestyle = '-'
            # linestyle = '--'
            ax.plot(t_array, particles_counter_mat2_3d[i, j, :], color=color, linestyle=linestyle)

            # y_filter = savitzky_golay(particles_counter_mat2_3d[i, j, :], window_size=13, order=1, deriv=0, rate=1)
            # ax.plot(t_array, y_filter, color='k', linewidth=3, linestyle='-')

        # if do_fit:
        #     # fit to exponential decay
        #     if j == i:
        #         def exp_decay(t, nu):
        #             return np.exp(-nu * t)
        #         p0 = (1.0)
        #         # def exp_decay(t, nu, b):
        #         #     return np.exp(-nu * t) + b
        #         # p0 = (1.0, 0.5)
        #         # params, cv = scipy.optimize.curve_fit(exp_decay, t_array[inds_t_array],
        #         #                                       particles_counter_mat2_3d[i, j, inds_t_array], p0)
        #         # nu = params[0]
        #         # # b = params[1]
        #         # nu_decay_list += [nu]
        #         # nu_mat[i, j] = nu
        #         # y_fit = exp_decay(t_array, params[0])
        #         # # y_fit = exp_decay(t_array, nu, b)
        #         # label = '$\\nu=$' + '{:.2f}'.format(nu) + '$s^{-1} \\cdot l/v_{th}$'
        #         # ax.plot(t_array, y_fit,
        #         #         # color=colors[j], linewidth=2, linestyle='--'
        #         #         color=color, linewidth=2, linestyle=':',
        #         #         label=label,
        #         #         )

    # ax.set_title('bin #' + str(i + 1))
    ax.set_title('bin #' + str(i + 1)
                 + ', $\\theta \\in$[' + '{:.1f}'.format(theta_bins_min_list[i])
                 + ',' + '{:.1f}'.format(theta_bins_max_list[i]) + ']')
    ax.set_xlabel('$t \\cdot v_{th} / l$')
    ax.grid(True)

# fit the exponential saturation
if do_fit:
    for i in range(N_theta):
        # if i == 0 or i == N_theta - 1:
        ax = axs3.ravel()[i]
        for j in range(N_theta):
            # if i != j:
            # if i != j and np.any(particles_counter_mat_3d[i, j, inds_t_array]):
            # if j in [i - 1, i + 1]:
            # if i == 0 or i == N_theta - 1:
            if j in [i - 1, i + 1] and np.any(particles_counter_mat2_3d[i, j, inds_t_array]):
                # if j in [i - 2, i - 1, i + 1, i + 2]:
                if j == i - 1:
                    color = 'r'
                elif j == i + 1:
                    color = 'b'

                # use the beginning rise, only smear the oscilations in the end
                # inds_t_rise = range(2)
                # inds_t_rise = range(3)
                # n_early = particles_counter_mat2_3d[i, j, inds_t_rise]
                # inds_t_saturation = range(10, 21)
                # # inds_t_saturation = range(7, 21)
                # n_late = t_array[inds_t_saturation] * 0 + np.mean(particles_counter_mat2_3d[i, j, inds_t_saturation])
                # inds_t_fit = [i for i in inds_t_rise] + [i for i in inds_t_saturation]
                # t_for_fit = t_array[inds_t_fit]
                # n_for_fit = np.append(n_early, n_late)
                # # t_for_fit = t_array[inds_t_rise]
                # # n_for_fit = n_early
                # ax.plot(t_for_fit, n_for_fit, color=color, linestyle='None', marker='o', linewidth=3, label='points for fit')

                # def exp_saturation(t, nu):
                #     return nu / nu_decay_list[i] * (1.0 - np.exp(-nu_decay_list[i] * t))
                # p0 = (0.01)

                # def exp_saturation(t, nu2, b):
                #     return b * (1 - np.exp(-nu2 * t))
                # p0 = (10, 0.1)
                # params, cv = scipy.optimize.curve_fit(exp_saturation, t_for_fit, n_for_fit, p0)
                # nu2 = params[0]
                # b = params[1]
                # nu = nu2 * b

                # def exp_saturation(t, nu2, b, c):
                #     return c - b * np.exp(-nu2 * t)
                # p0 = (0.5, 1.0, 1.0)
                # params, cv = scipy.optimize.curve_fit(exp_saturation, t_for_fit, n_for_fit, p0)
                # nu2 = params[0]
                # b = params[1]
                # c = params[2]
                # nu = nu2 * b

                # def exp_saturation(t, nu):
                #     return nu * t
                # p0 = (0.5)
                # params, cv = scipy.optimize.curve_fit(exp_saturation, t_for_fit, n_for_fit, p0)
                # nu = params[0]

                # nu_mat[i, j] = nu
                # label = 'fit $\\nu=$' + '{:.2f}'.format(nu_mat[i, j]) + '$s^{-1} \\cdot l/v_{th}$'
                #
                # # find the rise time to 90% saturation and use that for estimating the rate
                # t_search_array = np.linspace(0, np.max(t_array), 1000)
                # n_cutoff = 0.9 * b
                # ind_rise_to_sat = np.where(exp_saturation(t_search_array, nu2, b) >= n_cutoff)[0][0]
                # t_cutoff = t_search_array[ind_rise_to_sat]
                # ax.scatter(t_cutoff, n_cutoff, marker='s', color='None', edgecolor=color)
                # rate_estimate = n_cutoff / t_cutoff
                # nu_mat[i, j] = rate_estimate
                # label += ', rateest=' + '{:.2f}'.format(rate_estimate)

                # ax.plot(t_array,
                #         # exp_saturation(t_array, nu2, b, c),
                #         exp_saturation(t_array, nu2, b),
                #         # exp_saturation(t_array, 177, 0.19),
                #         # exp_saturation(t_array, nu),
                #         # color=colors[j], linewidth=2, linestyle='--'
                #         color=color, linewidth=2, linestyle=':',
                #         label=label,
                #         )

                # ax.plot(t_array,
                #         t_array * nu,
                #         color=color, linewidth=1, linestyle=':',
                #         )

                ## calculate the saturation value to estimate the rate
                # inds_t_saturation = range(7, 21)
                # inds_t_saturation = range(2, 3)
                inds_t_saturation = range(15, 31)
                # inds_t_saturation = range(len(t_array))
                saturation_value = np.mean(particles_counter_mat2_3d[i, j, inds_t_saturation])
                # the rate is the saturation value divided by the single cell pass time (1 in these units)
                saturation_rate = saturation_value / 1.0

                nu_mat[i, j] = saturation_rate
                label = 'fit $\\nu=$' + '{:.2f}'.format(nu_mat[i, j]) + '$\\cdot l/v_{th}$'
                # ax.plot(t_array, saturation_rate + 0 * t_array, color=color, linewidth=2, linestyle='--',
                #         label=label,
                #         )
                ax.hlines(saturation_rate, t_array[inds_t_saturation[0]], t_array[inds_t_saturation[-1]], color=color,
                          linewidth=2, linestyle='--',
                          label=label,
                          )
                ax.set_ylim([0, 1])
                ax.legend()

fig3.set_tight_layout({'pad': 0.5, 'rect': (0, 0, 1, 0.95)})

selectivity = nu_mat[0, 1] / nu_mat[2, 1]
# selectivity = nu_mat[0, 0] / nu_mat[2, 2]
title += ', selectivity=' + '{:.2f}'.format(selectivity)
fig3.suptitle(title)

## plot a heat map of all the rates

# annot = False
annot = True
annot_fontsize = 8
annot_fmt = '.1f'

# ind_t = 1
# rates_mat = (particles_counter_mat_3d[:, :, ind_t] - particles_counter_mat_3d[:, :, 0]) / t_array[ind_t]
# rates_mat = abs(rates_mat)
# # rates_mat = particles_counter_mat_3d[:, :, 1]
# fig3, ax3 = plt.subplots(figsize=(7, 6))
# sns.heatmap(rates_mat,
#             xticklabels=range(1, N_theta + 1), yticklabels=range(1, N_theta + 1),
#             # vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax3,
#             )
# # ax3.axes.invert_yaxis()
# ax3.set_ylabel('bin ini')
# ax3.set_xlabel('bin fin')
# ax3.set_title('$\\dot{N}(t) \\cdot l v_{th}$')

# fig4, ax4 = plt.subplots(figsize=(10, 8))
# sns.heatmap(nu_mat,
#             xticklabels=range(1, N_theta + 1), yticklabels=range(1, N_theta + 1),
#             # vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax4,
#             )
# # ax3.axes.invert_yaxis()
# ax4.set_ylabel('bin ini')
# ax4.set_xlabel('bin fin')
# ax4.set_title('fit $\\nu \\cdot l v_{th}$, for ' + title)
