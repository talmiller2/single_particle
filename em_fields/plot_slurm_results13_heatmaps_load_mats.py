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

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_cyclotron_angular_frequency

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
save_dir += '/set39_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'

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
m_curr = 2
# gas_name = 'DT_mix'
# m_curr = 2.5
# gas_name = 'tritium'
# m_curr = 3

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
N_rc = mat_dict['N_rc']
N_lc = mat_dict['N_lc']
N_cr = mat_dict['N_cr']
N_cl = mat_dict['N_cl']
# selectivity = mat_dict['selectivity']
selectivity = N_rc / N_lc
selectivity_trapped = N_cr / N_cl

alpha_const_omega_cyc0_right_list = []
alpha_const_omega_cyc0_left_list = []
vz_over_vth = 0.8
# vz_over_vth = 0.56
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

    # sns.lineplot(data=data, x='x', y='y', style=True, dashes=[(2, 2)], color=color, linewidth=3, )
    pass

    return


### PLOTS

annot = False
annot_fontsize = 8
annot_fmt = '.2f'

# yticklabels = alpha_loop_list
# ylabel = '$f_{\\omega}$'

_, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
q = Z_ion * settings['e']  # Coulomb
omega0 = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)
omega = alpha_loop_list * field_dict['omega_cyclotron']
yticklabels = ['{:.2f}'.format(w) for w in omega / omega0]
ylabel = '$\\omega / \\omega_{0,T}$'

if gas_name == 'deuterium':
    gas_name_shorthand = 'D'
if gas_name == 'tritium':
    gas_name_shorthand = 'T'

###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_rc
vmin = 0
vmax = 0.9
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], color='k')
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel(ylabel)
title = '$\\bar{N}_{rc}$'
title += ' (' + gas_name_shorthand + ')'
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(a)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)

###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_lc
# vmin = np.nanmin(y)
# vmax = np.nanmax(y)
vmin = 0
vmax = 0.9
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], color='k')
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel(ylabel)
title = '$\\bar{N}_{lc}$'
title += ' (' + gas_name_shorthand + ')'
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(b)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)

###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_cr
vmin = 0
vmax = 0.04
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], color='k')
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], color='k')
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel(ylabel)
title = '$\\bar{N}_{cr}$'
title += ' (' + gas_name_shorthand + ')'
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(c)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)

###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_cl
vmin = 0
vmax = 0.04
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], color='k')
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], color='k')
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
ax.set_ylabel(ylabel)
title = '$\\bar{N}_{cl}$'
title += ' (' + gas_name_shorthand + ')'
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(d)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)

# ###################
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = selectivity
# vmin = np.nanmin(y)
# vmax = np.nanmax(y)
# sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
#             vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax,
#             )
# ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], color='k')
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
# ax.set_ylabel(ylabel)
# ax.set_title('$\\bar{N}_{rc} / \\bar{N}_{lc}$')
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# text = '(e)'
# plt.text(0.15, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
#          horizontalalignment='right', verticalalignment='top', color='w',
#          transform=fig.axes[0].transAxes)
# ax.legend().set_visible(False)
#
#
# ###################
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = selectivity_trapped
# vmin = np.nanmin(y)
# vmax = np.nanmax(y)
# sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
#             vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax,
#             )
# ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i], color='k')
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i], color='k')
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
# ax.set_ylabel(ylabel)
# ax.set_title('$\\bar{N}_{cr} / \\bar{N}_{cl}$')
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# text = '(f)'
# plt.text(0.2, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
#          horizontalalignment='right', verticalalignment='top', color='w',
#          transform=fig.axes[0].transAxes)
# ax.legend().set_visible(False)


## save plots to file
save_dir = '../../../Papers/texts/paper2022/pics/'

# file_prefix = 'ERF_50kVm_'
# file_prefix = 'BRF_0.04T_'

# file_name = 'Nrc_heatmap_' + gas_name_shorthand
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = 'Nlc_heatmap_' + gas_name_shorthand
# beingsaved = plt.figure(2)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = 'Ncr_heatmap_' + gas_name_shorthand
# beingsaved = plt.figure(3)1
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = 'Ncl_heatmap_' + gas_name_shorthand
# beingsaved = plt.figure(4)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
