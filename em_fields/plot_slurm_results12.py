import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 10})


# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'


def plot_dist(y, ax, ylabel, color='b'):
    """
    Plot a subplot with the distribution of final state metric relative to initial angle
    """

    if 'z' in ylabel:
        label = '%(z/l>1)=' + '{:.2f}'.format(100.0 * len(np.where(y > 1)[0]) / len(y))
        label += ', %(z/l<0)=' + '{:.2f}'.format(100.0 * len(np.where(y < 0)[0]) / len(y))
    elif '\\Delta ( v_{\\perp} / v )' in ylabel:
        ### define selectivity
        right_LC_mean = np.mean(y[inds_right_LC])
        left_LC_mean = np.mean(y[inds_left_LC])
        right_trapped_mean = np.mean(y[inds_right_trapped])
        left_trapped_mean = np.mean(y[inds_left_trapped])
        right_mean = np.mean(y[inds_right])
        left_mean = np.mean(y[inds_left])
        selectivity_LC = right_LC_mean / left_LC_mean
        label = 'selectivity metrics: '
        label += 'RL(LC)=' + '{:.2f}'.format(selectivity_LC)
        selectivity_RL = right_mean / left_mean
        label += ', RL(tot)=' + '{:.2f}'.format(selectivity_RL)
        selectivity_Rwt = right_LC_mean * len(inds_right_LC) / (right_trapped_mean * len(inds_right_trapped))
        selectivity_Lwt = left_LC_mean * len(inds_left_LC) / (left_trapped_mean * len(inds_left_trapped))
        label += ', Rwt=' + '{:.2f}'.format(selectivity_Rwt)
        label += ', Lwt=' + '{:.2f}'.format(selectivity_Lwt)


    elif '\\Delta v / v_{th}' in ylabel:
        all_mean = np.mean(y)
        label = 'mean=' + '{:.2f}'.format(all_mean)
    else:
        label = None

    ax.scatter(theta0, y, color=color, alpha=0.3, label=label)
    # ax.scatter(theta_ini, y, color=color, alpha=0.5, label=label)

    ymin = np.min(y)
    ymax = np.max(y)

    ax.vlines(theta_LC, ymin, ymax, color='k', linestyle='--')
    ax.vlines(180 - theta_LC, ymin, ymax, color='k', linestyle='--')
    # ax.vlines(-theta_LC, ymin, ymax, color='k', linestyle='--')
    if '\\Delta ( v_{\\perp} / v )' in ylabel:
        ax.hlines(right_LC_mean, 0, theta_LC, color=color, linestyle='-', linewidth=2)
        ax.hlines(left_LC_mean, 180 - theta_LC, 180, color=color, linestyle='-', linewidth=2)
        ax.hlines(right_trapped_mean, theta_LC, 90, color='k', linestyle='-', linewidth=1)
        ax.hlines(left_trapped_mean, 90, 180 - theta_LC, color='k', linestyle='-', linewidth=1)
    elif '\\Delta v / v_{th}' in ylabel:
        ax.hlines(all_mean, 0, 180, color=color, linestyle='-', linewidth=2)
    # ax.set_xlabel('$\\theta$ [deg]')
    ax.set_xlabel('$\\theta_0$ [deg]')
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    if label is not None:
        ax.legend()
    ax.grid(True)

    # y_binned_stats = scipy.stats.binned_statistic(theta0, y, statistic='mean', bins=theta_bins)
    # y_std_binned_stats = scipy.stats.binned_statistic(theta0, y, statistic='std', bins=theta_bins)
    # plt.errorbar(theta_bins, y_binned_stats, yerr=y_std_binned_stats, label='stats', color='r')

    return


plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set26_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set28_B0_1T_l_10m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set29_B0_1T_l_3m_Post_Rm_2_first_cell_center_crossing/'
# save_dir += '/set30_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
save_dir += '/set31_B0_1T_l_3m_Post_Rm_3_intervals/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
E_RF_kVm = 25  # kV/m
# E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

# RF_type = 'magnetic_transverse'
B_RF = 0.05  # T
# B_RF = 0.1  # T

# use_RF = True
use_RF = False

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
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
# beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

alpha_loop_list = np.round(np.linspace(0.8, 1.0, 11), 2)  # set31
beta_loop_list = np.round(np.linspace(-10, 0, 11), 2)

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
# alpha = 0.8
alpha = 0.82
# alpha = 0.85
# alpha = 0.86
# alpha = 0.9
# alpha = 0.92
# alpha = 0.94
# alpha = 0.95
# alpha = 0.97
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
# beta = -4.5
# beta = -5.0
# beta = -7.5
beta = -8.0
# beta = -9.0
# beta = -10.0
# beta = 0.5
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
    set_name = 'r0_' + str(r_0) + '_' + set_name

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

normalize_by_tfin = True
# normalize_by_tfin = False

# fig, (axs) = plt.subplots(2, 2, figsize=(10,6))
# fig, (axs) = plt.subplots(3, 2, figsize=(10, 9))
# fig, (axs) = plt.subplots(3, 2, figsize=(12, 7))
fig, (axs) = plt.subplots(3, 1, figsize=(16, 9))

number_of_time_intervals = 3
# number_of_time_intervals = data_dict['t'].shape[1]

# TODO: split the populatin into several pieces to calculate how the number in each bin changes with time


# colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
# colors = ['b', 'g', 'r']
colors = ['r', 'g', 'b']

for ind_t in range(number_of_time_intervals):
    inds_particles = range(data_dict['t'].shape[0])
    # inds_particles = [0, 1, 2]
    # inds_particles = range(1001)

    t_ini = data_dict['t'][inds_particles, 0]
    v0 = data_dict['v'][inds_particles, ind_t]
    vz0 = data_dict['v_axial'][inds_particles, ind_t]
    vt0 = data_dict['v_transverse'][inds_particles, ind_t]
    theta0 = np.mod(360 / (2 * np.pi) * np.arctan(vt0 / vz0), 180)

    # dt = data_dict['t'][inds_particles, ind_t + 1] - data_dict['t'][inds_particles, ind_t]
    dt = data_dict['t'][inds_particles, ind_t + 1] - t_ini
    z = data_dict['z'][inds_particles, ind_t + 1]
    v = data_dict['v'][inds_particles, ind_t + 1]
    vt = data_dict['v_transverse'][inds_particles, ind_t + 1]
    vz = data_dict['v_axial'][inds_particles, ind_t + 1]
    theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)

    theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))

    theta_ini = theta0

    inds_right_LC = np.where(theta0 <= theta_LC)[0]
    inds_left_LC = np.where(theta0 >= 180 - theta_LC)[0]
    # print('len inds_right_LC=' + str(len(inds_right_LC)) + ', inds_left_LC=' + str(len(inds_left_LC)))
    # inds_right_half_LC = np.where(theta0 <= theta_LC / 2.0)[0]
    # inds_left_half_LC = np.where(theta0 >= 180 - theta_LC / 2.0)[0]
    # print('len inds_right_half_LC=' + str(len(inds_right_half_LC)) + ', inds_left_half_LC=' + str(len(inds_left_half_LC)))
    inds_right_trapped = [i for i, t in enumerate(theta0) if t > theta_LC and t < 90]
    inds_left_trapped = [i for i, t in enumerate(theta0) if t > 90 and t < 180 - theta_LC]
    inds_right = [i for i, t in enumerate(theta0) if t < 90]
    inds_left = [i for i, t in enumerate(theta0) if t > 90 and t < 180]

    # theta_bins = np.linspace(-90, 90, 31)
    # theta_bins = np.linspace(-90, 90, 31) + 180 / 30.0
    # theta_bin_width = theta_bins[1] - theta_bins[0]
    # theta_bin_width = 3
    # theta_bin_width = 5
    theta_bin_width = 10
    # theta_bins = np.arange(-90, 90, theta_bin_width)
    # theta_bins = np.delete(theta_bins, np.where(theta_bins == 0))
    # theta_bins = np.arange(theta_bin_width / 2.0, 90, theta_bin_width)
    # theta_bins = np.append(-np.flip(theta_bins), theta_bins)
    theta_bins = np.arange(theta_bin_width / 2.0, 180, theta_bin_width)

    color = colors[ind_t]

    # ax = axs[0, 0]
    # if ind_t == 0:
    #     ax.scatter(vz0 / settings['v_th'], vt0 / settings['v_th'], label=str(ind_t), alpha=0.3, color='k')
    # ax.scatter(vz / settings['v_th'], vt / settings['v_th'], label=str(ind_t + 1), alpha=0.3, color=color)
    # ax.set_xlabel('$v_z / v_{th}$')
    # ax.set_ylabel('$v_{\\perp} / v_{th}$')
    # ax.set_title(title)
    # ax.legend()
    # ax.grid(True)

    y = (v - v0) / settings['v_th']
    ylabel = '$\\Delta v / v_{th} $'
    if normalize_by_tfin is True:
        y /= dt / (settings['l'] / settings['v_th'])
        ylabel = '$\\Delta v / v_{th}  / ( \\Delta t v_{th} / l)$'
    # ax = axs[0, 1]
    ax = axs[0]
    plot_dist(y, ax, ylabel=ylabel, color=color)
    ax.set_ylim(-0.2, 0.2)
    ax.set_title(title)

    # y = (vt - vt0) / settings['v_th']
    # ylabel = '$\\Delta v_{\\perp} / v_{th}$'
    # if normalize_by_tfin is True:
    #     y /= dt / (settings['l'] / settings['v_th'])
    #     ylabel = '$\\Delta v_{\\perp} / v_{th} / (\\Delta t v_{th} / l)$'
    # ax = axs[1, 0]
    # plot_dist(y, ax, ylabel=ylabel)

    # y = theta
    # ylabel = '$\\theta$ [deg]'
    # # if normalize_by_tfin is True:
    # #     y /= dt / (settings['l'] / settings['v_th'])
    # #     ylabel = '$\\theta_{fin} / (\\Delta t v_{th} / l)$'
    # ax = axs[1, 0]
    # plot_dist(y, ax, ylabel=ylabel, color=color)
    # # if normalize_by_tfin is False:
    # ax.hlines(theta_LC, 0, 180, color='k', linestyle='--')
    # ax.hlines(180 - theta_LC, 0, 180, color='k', linestyle='--')

    y = (vt / v - vt0 / v0)
    ylabel = '$\\Delta ( v_{\\perp} / v )$'
    if normalize_by_tfin is True:
        y /= dt / (settings['l'] / settings['v_th'])
        ylabel = '$\\Delta ( v_{\\perp} / v ) / (\\Delta t v_{th} / l)$'
    # ax = axs[1, 1]
    ax = axs[1]
    plot_dist(y, ax, ylabel=ylabel, color=color)
    ax.set_ylim(-0.2, 0.2)

    # ax = axs[2, 0]
    ax = axs[2]
    y = z / settings['l']
    plot_dist(y, ax, ylabel='$z_{fin}/l$', color=color)

    # ax = axs[2, 1]
    # y = dt / (settings['l'] / settings['v_th'])
    # plot_dist(y, ax, ylabel='$\\Delta t v_{th} / l$', color=color)

fig.set_tight_layout(0.5)
# plt.tight_layout()
