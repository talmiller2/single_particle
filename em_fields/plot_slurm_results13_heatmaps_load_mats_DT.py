import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import loadmat

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
save_dir += '/set37_B0_1T_l_1m_Post_Rm_3_intervals/'

save_dir_curr = save_dir + 'without_RF'
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

RF_type = 'electric_transverse'
# E_RF_kVm = 0.1 # kV/m
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 25  # kV/m
E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

# RF_type = 'magnetic_transverse'
# B_RF = 0.001  # T
# B_RF = 0.01  # T
# B_RF = 0.02  # T
B_RF = 0.04  # T
# B_RF = 0.05  # T
# B_RF = 0.1  # T


gas_name = 'deuterium'
# gas_name = 'DT_mix'
# gas_name = 'tritium'

set_name = 'compiled_'
if RF_type == 'electric_transverse':
    set_name += 'ERF_' + str(E_RF_kVm)
elif RF_type == 'magnetic_transverse':
    set_name += 'BRF_' + str(B_RF)
if gas_name != 'hydrogen':
    set_name += '_' + gas_name
save_file = save_dir + '/' + set_name + '.mat'

mat_dict = loadmat(save_file)
alpha_loop_list = mat_dict['alpha_loop_list'][0]
beta_loop_list = mat_dict['beta_loop_list'][0]
rate_R_1 = mat_dict['rate_R']
rate_L_1 = mat_dict['rate_L']
selectivity_1 = mat_dict['selectivity']

# gas_name = 'deuterium'
# gas_name = 'DT_mix'
gas_name = 'tritium'

set_name = 'compiled_'
if RF_type == 'electric_transverse':
    set_name += 'ERF_' + str(E_RF_kVm)
elif RF_type == 'magnetic_transverse':
    set_name += 'BRF_' + str(B_RF)
if gas_name != 'hydrogen':
    set_name += '_' + gas_name
save_file = save_dir + '/' + set_name + '.mat'

mat_dict_2 = loadmat(save_file)
rate_R_2 = mat_dict_2['rate_R']
rate_L_2 = mat_dict_2['rate_L']
selectivity_2 = mat_dict_2['selectivity']

alpha_const_omega_cyc0_right_list = []
alpha_const_omega_cyc0_left_list = []
vz_over_vth = 0.6
# offset = 1.0
# slope = 2 * np.pi / settings['l'] * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
# m_curr = 2
m_curr = 2.5
# m_curr = 3
offset = 2.5 / m_curr
slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
alpha_const_omega_cyc0_right_list += [offset + slope * beta_loop_list]
alpha_const_omega_cyc0_left_list += [offset - slope * beta_loop_list]


def plot_line_on_heatmap(x_heatmap, y_heatmap, y_line, color='w'):
    x_heatmap_normed = 0.5 + np.array(range(len(x_heatmap)))
    y_line_normed = (y_line - y_heatmap[0]) / (y_heatmap[-1] - y_heatmap[0]) * len(y_heatmap) - 0.5
    # sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    # ax_line = sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    data = pd.DataFrame({'x': x_heatmap_normed, 'y': y_line_normed})
    # data = pd.DataFrame({'x': x_heatmap_normed, 'y': y_line_normed, 'style': ['--' for _ in range(len(y_line_normed))]})
    # ax_line = sns.lineplot(data=data, x='x', y='y', ax=ax)
    # ax_line.lines[0].set_linestyle(linestyle)

    sns.lineplot(data=data, x='x', y='y', style=True, dashes=[(2, 2)], color=color, linewidth=3, )

    return


### PLOTS

annot = False
annot_fontsize = 8
annot_fmt = '.2f'

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = selectivity_1
vmin = np.nanmin(y)
vmax = np.nanmax(y)
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel('$f_{\\omega}$')
ax.set_title('$s = \\bar{N}_{rc} / \\bar{N}_{lc}$ (D)')
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(a)'
plt.text(0.15, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = selectivity_2
vmin = np.nanmin(y)
vmax = np.nanmax(y)
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel('$f_{\\omega}$')
ax.set_title('$s = \\bar{N}_{rc} / \\bar{N}_{lc}$ (T)')
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(b)'
plt.text(0.15, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = np.minimum(selectivity_1, selectivity_2)
vmin = np.nanmin(y)
vmax = np.nanmax(y)
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel('$f_{\\omega}$')
ax.set_title('$min(s_D,s_T)$')
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(c)'
plt.text(0.15, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = 100 * (1 - (selectivity_1 - selectivity_2) / (0.5 * (selectivity_1 + selectiv1ity_1)))
y = 1 - abs((selectivity_1 - selectivity_2) / (0.5 * (selectivity_1 + selectivity_1)))
# y = abs((selectivity_1 - selectivity_2))
# vmin = 0
# vmax = 1
vmin = 0.7
vmax = 1.3
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel('$f_{\\omega}$')
ax.set_title('$1 - |s_D-s_T|/0.5(s_D+s_T)$')
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(d)'
plt.text(0.15, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)

## save plots to file
save_dir = '../../../Papers/texts/paper2022/pics/'

# file_prefix = 'ERF_50kVm_'
# file_prefix = 'BRF_0.04T_'

# file_name = file_prefix + 'selectivity_heatmap_RF_parameters'
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = file_prefix + 'Nrc_heatmap_RF_parameters'
# beingsaved = plt.figure(2)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = file_prefix + 'Nlc_heatmap_RF_parameters'
# beingsaved = plt.figure(3)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
