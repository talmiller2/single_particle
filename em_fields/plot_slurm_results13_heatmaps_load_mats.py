import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_cyclotron_angular_frequency

plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 12})
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
# save_dir += '/set41_B0_1T_l_1m_Post_Rm_3_intervals_D_T_ERF_25/'
# save_dir += '/set45_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set46_B0_2T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'

save_dir_curr = save_dir + 'without_RF'
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

LC_ini_fraction = np.sin(np.arcsin(field_dict['Rm'] ** (-0.5)) / 2) ** 2
trapped_ini_fraction = 1 - 2 * LC_ini_fraction

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
# B_RF = 0.08  # T

# gas_name = 'deuterium'
# m_curr = 2
# gas_name = 'DT_mix'
# m_curr = 2.5
gas_name = 'tritium'
m_curr = 3

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'

with_RF_xy_corrections = True
induced_fields_factor = 1
# induced_fields_factor = 0.5
# induced_fields_factor = 0.1
# induced_fields_factor = 0.01
# induced_fields_factor = 0
# time_step_tau_cyclotron_divisions = 20
time_step_tau_cyclotron_divisions = 40
# time_step_tau_cyclotron_divisions = 80
# sigma_r0 = 0
sigma_r0 = 0.1

## save compiled data to file
set_name = 'compiled_'
if RF_type == 'electric_transverse':
    set_name += 'ERF_' + str(E_RF_kVm)
elif RF_type == 'magnetic_transverse':
    set_name += 'BRF_' + str(B_RF)
if 'set39' not in save_dir:
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
print(set_name)
save_file = save_dir + '/' + set_name + '.mat'

mat_dict = loadmat(save_file)
alpha_loop_list = mat_dict['alpha_loop_list'][0]
beta_loop_list = mat_dict['beta_loop_list'][0]
N_rc = mat_dict['N_rc']
N_lc = mat_dict['N_lc']
N_cr = mat_dict['N_cr']
N_cl = mat_dict['N_cl']
selectivity = mat_dict['selectivity']
# selectivity = N_rc / N_lc
# print('np.max(selectivity))', np.max(selectivity))
selectivity_trapped = N_cr / N_cl

alpha_const_omega_cyc0_right_list = []
alpha_const_omega_cyc0_right_list_v2 = []
alpha_const_omega_cyc0_left_list = []
# vz_over_vth = 0.8
# vz_over_vth = 0.56
vz_over_vth = 1.025  # mean of vz in loss cone
# vz_over_vth = 1.18 # analytic result for mean of vz in loss cone
offset = 2.5 / m_curr
slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']

# # define v_th ref
# _, _, mi, _, Z_ion = define_plasma_parameters(gas_name=gas_name)
# v_th_ref = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])
# q = Z_ion * settings['e']  # Coulomb
# omega_cyc = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)
# slope = 2 * np.pi * vz_over_vth * v_th_ref / omega_cyc

alpha_const_omega_cyc0_right_list += [offset + slope * beta_loop_list]
alpha_const_omega_cyc0_left_list += [offset - slope * beta_loop_list]


def plot_line_on_heatmap(x_heatmap, y_heatmap, y_line, color='k'):
    x_heatmap_normed = 0.5 + np.array(range(len(x_heatmap)))
    y_line_normed = (y_line - y_heatmap[0]) / (y_heatmap[-1] - y_heatmap[0]) * len(y_heatmap) - 0.5
    # sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    # ax_line = sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    data = pd.DataFrame({'x': x_heatmap_normed, 'y': y_line_normed})
    # data = pd.DataFrame({'x': x_heatmap_normed, 'y': y_line_normed, 'style': ['--' for _ in range(len(y_line_normed))]})
    # ax_line = sns.lineplot(data=data, x='x', y='y', ax=ax)
    # ax_line.lines[0].set_linestyle(linestyle)
    ax_line = sns.lineplot(data=data, x='x', y='y', style=True, dashes=[(2, 2)], color=color, linewidth=3)
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

# cbar_kws_dict = {'format': '%.2f'}
# cbar_kws_dict = {'format': '%.3f'}
# cbar_kws_dict = {'format': '%.2f', 'ticks': [0, 0.22, 0.45, 0.67, 0.9]}
# cbar_kws_dict = {'format': '%.1f'}
cbar_kws_dict = None

# lower resolution for xticks and yticks

beta_loop_list_copy = copy.deepcopy(beta_loop_list)
yticklabels_copy = copy.deepcopy(yticklabels)
# beta_loop_list = []
# yticklabels = []
# for i in range(len(beta_loop_list)):
#     if np.mod(i, 2) == 0:
#         # yticklabels += [str(int(beta_loop_list[i]))]
#         # yticklabels += [str(yticklabels[i])]
#         yticklabels += [str(beta_loop_list[i])]
#         yticklabels += [str(alpha_loop_list[i])]
#     else:
#         yticklabels += ['']
#         yticklabels += ['']
beta_loop_list = beta_loop_list
yticklabels = yticklabels


###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_rc
# y = N_rc * LC_ini_fraction # TODO: testing absolute quantities
vmin = 0
# vmax = 1.0
vmax = np.max(y)
sns.heatmap(y.T,
            xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            # cbar=False,
            cbar_kws=cbar_kws_dict,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
    # plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list_v2[i], color='w')
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# axes_label_size = 20
axes_label_size = 14
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
ax.set_ylabel(ylabel, fontsize=axes_label_size)
title = '$\\bar{N}_{rc}$'
title += ' (' + gas_name_shorthand + ')'
title += ', max=' + '{:.3f}'.format(np.max(y))
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
if gas_name == 'deuterium':
    text = '(a)'
else:
    text = '(e)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)
# ax.get_xaxis().set_visible(False)

###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_lc
# vmin = np.nanmin(y)
# vmax = np.nanmax(y)
vmin = 0
# vmax = 0.9
vmax = 1.0
# if RF_type == 'magnetic_transverse':
#     vmax = 0.6
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            cbar_kws=cbar_kws_dict,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
ax.set_ylabel(ylabel, fontsize=axes_label_size)
title = '$\\bar{N}_{lc}$'
title += ' (' + gas_name_shorthand + ')'
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
if gas_name == 'deuterium':
    text = '(b)'
else:
    text = '(f)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

###################

# cbar_kws_dict = {'format': '%.2f', 'ticks': [0, 0.01, 0.02, 0.03, 0.04]}
cbar_kws_dict = None

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_cr
# y = N_cr * trapped_ini_fraction # TODO: testing absolute quantities
vmin = 0
# vmax = 0.16
vmax = np.max(y)
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            # cbar=False,
            cbar_kws=cbar_kws_dict,
            )
ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
ax.set_ylabel(ylabel, fontsize=axes_label_size)
# if gas_name == 'tritium':
#     ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# else:
#     ax.get_xaxis().set_visible(False)
title = '$\\bar{N}_{cr}$'
title += ' (' + gas_name_shorthand + ')'
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
if gas_name == 'deuterium':
    text = '(c)'
else:
    text = '(g)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)

###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = N_cl
vmin = 0
# vmax = 0.04
vmax = 0.16
sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            cbar_kws=cbar_kws_dict,
            )
ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
ax.set_ylabel(ylabel, fontsize=axes_label_size)
# if gas_name == 'tritium':
#     ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# else:
#     ax.get_xaxis().set_visible(False)
# ax.set_ylabel(ylabel, fontsize=axes_label_size)
title = '$\\bar{N}_{cl}$'
title += ' (' + gas_name_shorthand + ')'
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
if gas_name == 'deuterium':
    text = '(d)'
else:
    text = '(h)'
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)
# ax.get_yaxis().set_visible(False)

###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = selectivity
vmin = np.nanmin(y)
vmax = np.nanmax(y)
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
# ax.set_title('$\\bar{N}_{rc} / \\bar{N}_{lc}$')
# ax.set_title('selectivity $\\bar{N}_{rc} / \\bar{N}_{lc}$')
ax.set_title('selectivity $\\bar{N}_{rc} / \\bar{N}_{lc}$, max=' + '{:.3f}'.format(np.max(y)))
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
text = '(e)'
plt.text(0.15, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='right', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)

###################
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
# # ax.set_title('$\\bar{N}_{cr} / \\bar{N}_{cl}$')
# ax.set_title('selectivity $\\bar{N}_{cr} / \\bar{N}_{cl}$')
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# text = '(f)'
# plt.text(0.2, 0.95, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
#          horizontalalignment='right', verticalalignment='top', color='w',
#          transform=fig.axes[0].transAxes)
# ax.legend().set_visible(False)


###################
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = N_rc * LC_ini_fraction / (N_cr * trapped_ini_fraction) # TODO: testing absolute quantities
# # y = 1 / y
# # vmin = 1.0
# # vmax = 1.5
# vmin = np.min(y)
# vmax = np.max(y)
# sns.heatmap(y.T,
#             xticklabels=beta_loop_list, yticklabels=yticklabels,
#             vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax,
#             # cbar=False,
#             cbar_kws=cbar_kws_dict,
#             )
# ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
# # ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# # axes_label_size = 20
# axes_label_size = 14
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# ax.set_ylabel(ylabel, fontsize=axes_label_size)
# title = '$\\bar{N}_{rc}/\\bar{N}_{cr}$'
# title += ' absolute'
# title += ' (' + gas_name_shorthand + ')'
# # title += ', max=' + '{:.3f}'.format(np.max(y))
# ax.set_title(title, fontsize=20)
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
#          horizontalalignment='left', verticalalignment='top', color='w',
#          transform=fig.axes[0].transAxes)
# ax.legend().set_visible(False)
# # ax.get_xaxis().set_visible(False)
#
# ###################
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = N_rc * LC_ini_fraction / (N_cr * trapped_ini_fraction) # TODO: testing absolute quantities
# y = 1 / y
# # vmin = 1.0
# # vmax = 1.5
# vmin = np.min(y)
# vmax = np.max(y)
# sns.heatmap(y.T,
#             xticklabels=beta_loop_list, yticklabels=yticklabels,
#             vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax,
#             # cbar=False,
#             cbar_kws=cbar_kws_dict,
#             )
# ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
# # ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# # axes_label_size = 20
# axes_label_size = 14
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# ax.set_ylabel(ylabel, fontsize=axes_label_size)
# title = '$\\bar{N}_{cr}/\\bar{N}_{rc}$'
# title += ' absolute'
# title += ' (' + gas_name_shorthand + ')'
# # title += ', max=' + '{:.3f}'.format(np.max(y))
# ax.set_title(title, fontsize=20)
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
#          horizontalalignment='left', verticalalignment='top', color='w',
#          transform=fig.axes[0].transAxes)
# ax.legend().set_visible(False)
# # ax.get_xaxis().set_visible(False)


###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = (N_rc * LC_ini_fraction - N_cr * trapped_ini_fraction) / LC_ini_fraction  # TODO: testing absolute quantities
# vmin = 1.0
# vmax = 1.5
vmin = np.min(y)
vmax = np.max(y)
sns.heatmap(y.T,
            xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            # cbar=False,
            cbar_kws=cbar_kws_dict,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# axes_label_size = 20
axes_label_size = 14
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
ax.set_ylabel(ylabel, fontsize=axes_label_size)
title = '$(N_{rc}-N_{cr})/N_{cone}$'
title += ' (' + gas_name_shorthand + ')'
# title += ', max=' + '{:.3f}'.format(np.max(y))
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)
# ax.get_xaxis().set_visible(False)


###################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
y = (N_lc * LC_ini_fraction - N_cl * trapped_ini_fraction) / LC_ini_fraction  # TODO: testing absolute quantities
# vmin = 1.0
# vmax = 1.5
vmin = np.min(y)
vmax = np.max(y)
sns.heatmap(y.T,
            xticklabels=beta_loop_list, yticklabels=yticklabels,
            vmin=vmin, vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax,
            # cbar=False,
            cbar_kws=cbar_kws_dict,
            )
ax.axes.invert_yaxis()
for i in range(len(alpha_const_omega_cyc0_right_list)):
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
# axes_label_size = 20
axes_label_size = 14
ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
ax.set_ylabel(ylabel, fontsize=axes_label_size)
title = '$(N_{lc}-N_{cl})/N_{cone}$'
title += ' (' + gas_name_shorthand + ')'
# title += ', max=' + '{:.3f}'.format(np.max(y))
ax.set_title(title, fontsize=20)
fig.set_tight_layout(0.5)
plt.yticks(rotation=0)
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='left', verticalalignment='top', color='w',
         transform=fig.axes[0].transAxes)
ax.legend().set_visible(False)
# ax.get_xaxis().set_visible(False)

###################
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = mat_dict['percent_ok']
# # print('percent_ok min=' + str(np.min(mat_dict['percent_ok'])) + ', max=' + str(np.max(mat_dict['percent_ok'])))
# vmin = np.nanmin(y)
# vmax = np.nanmax(y)
# # vmin = 90
# # vmax = 100
# sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
#             vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax,
#             )
# ax.axes.invert_yaxis()
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
# ax.set_ylabel(ylabel)
# ax.set_title('%ok')
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# ax.legend().set_visible(False)

###################
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = mat_dict['dE_mean']
# # vmin = np.nanmin(y)
# # vmax = np.nanmax(y)
# vmin = np.nanmin(1)
# vmax = np.nanmax(2)
# sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
#             vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax,
#             )
# ax.axes.invert_yaxis()
# for i in range(len(alpha_const_omega_cyc0_right_list)):
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_right_list[i])
#     plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_cyc0_left_list[i])
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
# ax.set_ylabel(ylabel)
# ax.set_title('mean($E_f/E_i$)')
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# ax.legend().set_visible(False)

###################
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# y = mat_dict['dE_std']
# vmin = np.nanmin(y)
# vmax = np.nanmax(y)
# sns.heatmap(y.T, xticklabels=beta_loop_list, yticklabels=yticklabels,
#             vmin=vmin, vmax=vmax,
#             annot=annot,
#             annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
#             ax=ax,
#             )
# ax.axes.invert_yaxis()
# ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$')
# ax.set_ylabel(ylabel)
# ax.set_title('std($E_f/E_i$)')
# fig.set_tight_layout(0.5)
# plt.yticks(rotation=0)
# ax.legend().set_visible(False)



## save plots to file
save_dir = '../../../Papers/texts/paper2022/pics/'

# file_name = 'Nrc_heatmap_' + gas_name_shorthand
# if RF_type == 'magnetic_transverse':
#     file_name = 'BRF_' + file_name
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = 'Nlc_heatmap_' + gas_name_shorthand
# if RF_type == 'magnetic_transverse':
#     file_name = 'BRF_' + file_name
# beingsaved = plt.figure(2)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = 'Ncr_heatmap_' + gas_name_shorthand
# if RF_type == 'magnetic_transverse':
#     file_name = 'BRF_' + file_name
# beingsaved = plt.figure(3)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = 'Ncl_heatmap_' + gas_name_shorthand
# if RF_type == 'magnetic_transverse':
#     file_name = 'BRF_' + file_name
# beingsaved = plt.figure(4)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
