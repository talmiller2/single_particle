import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns

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
save_dir += '/set30_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'

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
# E_RF_kVm = 25  # kV/m
E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

# RF_type = 'magnetic_transverse'
B_RF = 0.05  # T
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

alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

# vz_over_vth_list = [0.5, 1.0, 1.5]
vz_over_vth_list = [0.5, 0.75, 1.0, 1.25, 1.5]
alpha_const_omega_cyc0_right_list = []
alpha_const_omega_cyc0_left_list = []
for vz_over_vth in vz_over_vth_list:
    alpha_const_omega_cyc0_right_list += [1.0 + beta_loop_list * 2 * np.pi / settings['l']
                                          * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']]
    alpha_const_omega_cyc0_left_list += [1.0 - beta_loop_list * 2 * np.pi / settings['l']
                                         * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']]

delta_v = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
selectivity_LC = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
# right_LC_trapping_metric = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
# left_LC_trapping_metric = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
selectivity_Rwt = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
selectivity_Lwt = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])
selectivity_RL = np.nan * np.zeros([len(beta_loop_list), len(alpha_loop_list)])


def plot_line_on_heatmap(x_heatmap, y_heatmap, y_line, ax=None, color='b', linewidth=2):
    x_heatmap_normed = 0.5 + np.array(range(len(x_heatmap)))
    y_line_normed = (y_line - y_heatmap[0]) / (y_heatmap[-1] - y_heatmap[0]) * len(y_heatmap) - 0.5
    sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    return


for ind_beta, beta in enumerate(beta_loop_list):
    for ind_alpha, alpha in enumerate(alpha_loop_list):

        # try:

        # if use_RF is False:
        #     title = 'without RF'
        # elif RF_type == 'electric_transverse':
        #     title = '$E_{RF}$=' + str(E_RF_kVm) + 'kV/m, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
        # elif RF_type == 'magnetic_transverse':
        #     title = '$B_{RF}$=' + str(B_RF) + 'T, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
        # print(title)

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
        # settings_file = save_dir + 'settings.pickle'
        # with open(settings_file, 'rb') as fid:
        #     settings = pickle.load(fid)
        # field_dict_file = save_dir + 'field_dict.pickle'
        # with open(field_dict_file, 'rb') as fid:
        #     field_dict = pickle.load(fid)

        for key in data_dict.keys():
            data_dict[key] = np.array(data_dict[key])

        dt = data_dict['t'][:, 1] - data_dict['t'][:, 0]
        z = data_dict['z'][:, 1]
        v = data_dict['v'][:, 1]
        v0 = data_dict['v'][:, 0]
        vt = data_dict['v_transverse'][:, 1]
        vt0 = data_dict['v_transverse'][:, 0]
        vz = data_dict['v_axial'][:, 1]
        vz0 = data_dict['v_axial'][:, 0]
        # theta0 = 360 / (2 * np.pi) * np.arcsin(vt0 / v0) * np.sign(vz0)
        theta0 = np.mod(360 / (2 * np.pi) * np.arctan(vt0 / vz0), 180)

        theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))

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

        normalize_by_tfin = True
        # normalize_by_tfin = False

        # y = (vt - vt0) / settings['v_th']
        # ylabel_delta_v = '$\\Delta v_{\\perp} / v_{th}$'
        # if normalize_by_tfin is True:
        #     y /= dt / (settings['l'] / settings['v_th'])
        #     ylabel_delta_v = '$\\Delta v_{\\perp} / v_{th} / (t_{fin} v_{th} / l)$'
        # ylabel_delta_v = 'median of ' + ylabel_delta_v
        # delta_v[ind_beta, ind_alpha] = np.percentile(y, 50)

        y = (v - v0) / settings['v_th']
        ylabel_delta_v = '$\\Delta v / v_{th} $'
        if normalize_by_tfin is True:
            y /= dt / (settings['l'] / settings['v_th'])
            ylabel_delta_v = '$\\Delta v / v_{th}  / (t_{fin} v_{th} / l)$'
        # ylabel_delta_v = 'median of ' + ylabel_delta_v
        # delta_v[ind_beta, ind_alpha] = np.percentile(y, 50)
        ylabel_delta_v = 'mean of ' + ylabel_delta_v
        delta_v[ind_beta, ind_alpha] = np.mean(y)

        y = (vt / v - vt0 / v0)
        ylabel_selectivity_trapping_metric = '$\\Delta ( v_{\\perp} / v )$'
        if normalize_by_tfin is True:
            y /= dt / (settings['l'] / settings['v_th'])
            ylabel_selectivity_trapping_metric = '$\\Delta ( v_{\\perp} / v ) / (t_{fin} v_{th} / l)$'
        ylabel_selectivity_LC = 'mean rightLC / leftLC of ' + ylabel_selectivity_trapping_metric
        ylabel_selectivity_Rwt = 'weighted mean rightLC / right-trapped of ' + ylabel_selectivity_trapping_metric
        ylabel_selectivity_Lwt = 'weighted mean leftLC / left-trapped of ' + ylabel_selectivity_trapping_metric
        ylabel_selectivity_RL = 'mean right / left of ' + ylabel_selectivity_trapping_metric

        right_LC_mean = np.mean(y[inds_right_LC])
        left_LC_mean = np.mean(y[inds_left_LC])
        right_trapped_mean = np.mean(y[inds_right_trapped])
        left_trapped_mean = np.mean(y[inds_left_trapped])
        right_mean = np.mean(y[inds_right])
        left_mean = np.mean(y[inds_left])
        selectivity_LC[ind_beta, ind_alpha] = right_LC_mean / left_LC_mean
        selectivity_Rwt[ind_beta, ind_alpha] = right_LC_mean * len(inds_right_LC) / (
                    right_trapped_mean * len(inds_right_trapped))
        selectivity_Lwt[ind_beta, ind_alpha] = left_LC_mean * len(inds_left_LC) / (
                    left_trapped_mean * len(inds_left_trapped))
        selectivity_RL[ind_beta, ind_alpha] = right_mean / left_mean

        # except:
        #     print('failed on ' + set_name)

### PLOTS

# fig, (axs) = plt.subplots(2, 2, figsize=(16, 10))
# fig, (axs) = plt.subplots(2, 3, figsize=(12, 8))
fig, (axs) = plt.subplots(2, 3, figsize=(16, 9))
annot_fontsize = 8

ax = axs[0, 0]
# plt.figure(1)
# plt.subplot(fignum=1)
log_delta_v = np.log(delta_v)
# vmin = 0
vmin = np.min(log_delta_v)
# vmax = np.nanmax(delta_v)
vmax = np.max(log_delta_v)
# vmax = 0.1
# vmax = 0.05
# vmax = 5
sns.heatmap(log_delta_v.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            # annot=True,
            # annot_kws={"fontsize": annot_fontsize},
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
# ax.set_title(ylabel_delta_v)
ax.set_title('log of ' + ylabel_delta_v)
# plt.tight_layout(pad=0.5)


# plt.figure(2)
ax = axs[0, 1]
# fig, ax = plt.subplots(figsize=(8, 8))
vmin = 0
# vmax = np.max(selectivity_LC)
# vmax = 3
# vmax = 5
# vmax = 10
vmax = 30
# vmax = 15
sns.heatmap(selectivity_LC.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            # annot=True,
            annot_kws={"fontsize": annot_fontsize},
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
ax.set_title(ylabel_selectivity_LC)

#
# # plt.figure(3)
# ax = axs[1, 0]
# vmin = 0
# # vmax = np.max(right_LC_trapping_metric)
# # vmax = 5
# vmax = 0.5
# # vmax = 1
# sns.heatmap(right_LC_trapping_metric.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
#             vmin=vmin, vmax=vmax,
#             # annot=True,
#             # annot_kws={"fontsize": annot_fontsize},
#             ax=ax,
#             )
# ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], ax=ax, color='b',
#                          linewidth=2)
# ax.set_xlabel('$\\beta$')
# ax.set_ylabel('$\\alpha$')
# ax.set_title(ylabel_right_LC_trapping_metric)
# # plt.tight_layout(pad=0.5)


# # plt.figure(4)
# ax = axs[1, 1]
# vmin = 0
# # vmax = np.max(left_LC_trapping_metric)
# # vmax = 5
# vmax = 0.5
# # vmax = 1
# sns.heatmap(left_LC_trapping_metric.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
#             vmin=vmin, vmax=vmax,
#             # annot=True,
#             # annot_kws={"fontsize": annot_fontsize},
#             ax=ax,
#             )
# ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], ax=ax, color='b',
#                          linewidth=2)
# ax.set_xlabel('$\\beta$')
# ax.set_ylabel('$\\alpha$')
# ax.set_title(ylabel_left_LC_trapping_metric)


ax = axs[1, 0]
vmin = 0
vmax = 30
sns.heatmap(selectivity_Rwt.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            # annot=True,
            annot_kws={"fontsize": annot_fontsize},
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
ax.set_title(ylabel_selectivity_Rwt)

ax = axs[1, 1]
vmin = 0
vmax = 30
sns.heatmap(selectivity_Lwt.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            # annot=True,
            annot_kws={"fontsize": annot_fontsize},
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
ax.set_title(ylabel_selectivity_Lwt)

ax = axs[0, 2]
vmin = 0
vmax = 30
sns.heatmap(selectivity_RL.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            # annot=True,
            annot_kws={"fontsize": annot_fontsize},
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
ax.set_title(ylabel_selectivity_RL)

fig.set_tight_layout(0.5)
# plt.tight_layout(pad=0.5)
