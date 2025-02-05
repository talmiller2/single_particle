import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_cyclotron_angular_frequency
from em_fields.plot_functions import update_format_coord

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

# cmap = 'viridis'
# cmap = 'plasma'
# cmap = 'inferno'
cmap = 'coolwarm'
figsize = (6, 6)

axes_label_size = 14
title_fontsize = 16

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
# save_dir += '/set46_B0_2T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set49_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set51_B0_1T_l_1m_Post_Rm_6_intervals_D_T/'
# save_dir += '/set52_B0_1T_l_2m_Post_Rm_3_intervals_D_T/'
save_dir += '/set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

plot_rate_eqs_flux = True
mm_rate_eqs_main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/'
mm_rate_eqs_main_dir += '/runs/slurm_runs/set47_MM_Rm_10_ni_1e21_Ti_10keV_withRMF'


select_alpha_list = []
select_beta_list = []
set_name_list = []

####

# select_alpha_list += [0.64]
# select_beta_list += [-1.8]
# set_name_list += ['1']
#
# select_alpha_list += [0.7]
# select_beta_list += [-0.8]
# set_name_list += ['2']
#
# select_alpha_list += [1.06]
# select_beta_list += [-1.8]
# set_name_list += ['3']
#
# select_alpha_list += [1.12]
# select_beta_list += [1.4]
# set_name_list += ['4']
#
# select_alpha_list += [0.88]
# select_beta_list += [0]
# set_name_list += ['5']
#
# select_alpha_list += [1.0]
# select_beta_list += [-1.8]
# set_name_list += ['6']
#
# select_alpha_list += [0.88]
# select_beta_list += [-1.8]
# set_name_list += ['7']

####

# select_alpha_list += [1.3, 1.48, 1.12, 1.06, 1.0]
# select_beta_list += [0, 1.4, -1.0, -1.4, -1.8]
# set_name_list += ['1', '2', '3', '4', '5']
#
# select_alpha_list += [0.88, 1.06, 0.76, 0.7, 0.64]
# select_beta_list += [0, 1.4, -1.0, -1.4, -1.8]
# set_name_list += ['6', '7', '8', '9', '10']
#
# select_alpha_list += [0.82, 1.06]
# select_beta_list += [-1.8, 0]
# set_name_list += ['11', '12']

#######
# alpha_loop_list
# beta_loop_list

# select_alpha_list += [1.3, 1.12, 1.06, 1.0, 1.42, 1.48]
# select_beta_list += [0, -1.0, -1.4, -1.8, 0.8, 1.4]
# set_name_list += ['1', '2', '3', '4', '5', '6']
#
# select_alpha_list += [0.88, 0.76, 0.7, 0.64, 1.0, 1.06]
# select_beta_list += [0, -1.0, -1.4, -1.8, 0.8, 1.4]
# set_name_list += ['7', '8', '9', '10', '11', '12']
#
# select_alpha_list += [1.06, 0.94, 0.88, 0.82]
# select_beta_list += [0, -1.0, -1.4, -1.8]
# set_name_list += ['13', '14', '15', '16']


select_alpha_list += [1.54, 1.54]
select_beta_list += [1.4, 1.6]
set_name_list += ['1', '2']

# alpha_loop_list
# Out[54]:
# array([0.4 , 0.46, 0.52, 0.58, 0.64, 0.7 , 0.76, 0.82, 0.88, 0.94, 1.  ,
#        1.06, 1.12, 1.18, 1.24, 1.3 , 1.36, 1.42, 1.48, 1.54, 1.6 ])
# beta_loop_list
# Out[55]:
# array([-2. , -1.8, -1.6, -1.4, -1.2, -1. , -0.8, -0.6, -0.4, -0.2,  0. ,
#         0.2,  0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8,  2. ])

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
E_RF_kVm = 25  # [kV/m]
# E_RF_kVm = 50 # [kV/m]
# RF_type = 'magnetic_transverse'
B_RF = 0.02  # [T]
# B_RF = 0.04  # [T]
# B_RF = 0.08  # [T]

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'

induced_fields_factor = 1
# induced_fields_factor = 0.5
# induced_fields_factor = 0.1
# induced_fields_factor = 0.01
# induced_fields_factor = 0
time_step_tau_cyclotron_divisions = 50
# time_step_tau_cyclotron_divisions = 100
# sigma_r0 = 0
sigma_r0 = 0.05
# sigma_r0 = 0.1
radial_distribution = 'uniform'

# theta_type = 'sign_vz0'
theta_type = 'sign_vz'

# gas_name = 'deuterium'
# gas_name = 'DT_mix'
gas_name = 'tritium'

if gas_name == 'deuterium':
    gas_name_short = 'D'
elif gas_name == 'tritium':
    gas_name_short = 'T'
else:
    gas_name_short = 'DTmix'

# with_kr_correction = False
with_kr_correction = True

# label_1 = gas_name_short + ', wo/kr'
label_1 = gas_name_short + ', iff=1'

set_name = 'compiled_'
set_name += theta_type + '_'
if RF_type == 'electric_transverse':
    set_name += 'ERF_' + str(E_RF_kVm)
elif RF_type == 'magnetic_transverse':
    set_name += 'BRF_' + str(B_RF)
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
print(set_name)
save_file = save_dir + '/' + set_name + '.mat'

mat_dict_1 = loadmat(save_file)
alpha_loop_list = mat_dict_1['alpha_loop_list'][0]
beta_loop_list = mat_dict_1['beta_loop_list'][0]
N_rc_1 = mat_dict_1['N_rc']
N_lc_1 = mat_dict_1['N_lc']
N_cr_1 = mat_dict_1['N_cr']
N_cl_1 = mat_dict_1['N_cl']
N_rl_1 = mat_dict_1['N_rl']
N_lr_1 = mat_dict_1['N_lr']
percent_ok_1 = mat_dict_1['percent_ok']
E_ratio_mean_1 = mat_dict_1['E_ratio_mean']
selectivity_1 = N_rc_1 / N_lc_1
cone_escape_rate_1 = (N_rc_1 * LC_ini_fraction - N_cr_1 * trapped_ini_fraction) / LC_ini_fraction

if plot_rate_eqs_flux:
    num_cells = 50
    rate_eqs_state_file = mm_rate_eqs_main_dir + '/' + set_name + '_N_' + str(num_cells) + '.mat'
    mm_rate_eqs_mat_dict = loadmat(rate_eqs_state_file)
    cross_section_main_cell = 0.19634954084936207
    flux_lawson_ignition_piel = 2.908441211385411e+22
    rate_eqs_flux_1 = mm_rate_eqs_mat_dict['flux_mat'] * cross_section_main_cell / flux_lawson_ignition_piel

########


with_kr_correction = True
# with_kr_correction = False
induced_fields_factor = 1
# induced_fields_factor = 0

# label_2 = gas_name_short + ', w/kr'
label_2 = gas_name_short + ', iff=0'

set_name = 'compiled_'
set_name += theta_type + '_'
if RF_type == 'electric_transverse':
    set_name += 'ERF_' + str(E_RF_kVm)
elif RF_type == 'magnetic_transverse':
    set_name += 'BRF_' + str(B_RF)
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
print(set_name)
save_file = save_dir + '/' + set_name + '.mat'

mat_dict_2 = loadmat(save_file)
N_rc_2 = mat_dict_2['N_rc']
N_lc_2 = mat_dict_2['N_lc']
N_cr_2 = mat_dict_2['N_cr']
N_cl_2 = mat_dict_2['N_cl']
N_rl_2 = mat_dict_2['N_rl']
N_lr_2 = mat_dict_2['N_lr']
percent_ok_2 = mat_dict_2['percent_ok']
E_ratio_mean_2 = mat_dict_2['E_ratio_mean']
selectivity_2 = N_rc_2 / N_lc_2
cone_escape_rate_2 = (N_rc_2 * LC_ini_fraction - N_cr_2 * trapped_ini_fraction) / LC_ini_fraction

if plot_rate_eqs_flux:
    num_cells = 50
    rate_eqs_state_file = mm_rate_eqs_main_dir + '/' + set_name + '_N_' + str(num_cells) + '.mat'
    mm_rate_eqs_mat_dict = loadmat(rate_eqs_state_file)
    rate_eqs_flux_2 = mm_rate_eqs_mat_dict['flux_mat'] * cross_section_main_cell / flux_lawson_ignition_piel


def plot_line_on_heatmap(x_heatmap, y_heatmap, y_line, color='w'):
    x_heatmap_normed = 0.5 + np.array(range(len(x_heatmap)))
    y_line_normed = (y_line - y_heatmap[0]) / (y_heatmap[-1] - y_heatmap[0]) * len(y_heatmap) - 0.5
    # sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    # ax_line = sns.lineplot(x=x_heatmap_normed, y=y_line_normed, color=color, linewidth=linewidth, ax=ax)
    data = pd.DataFrame({'x': x_heatmap_normed, 'y': y_line_normed})
    # data = pd.DataFrame({'x': x_heatmap_normed, 'y': y_line_normed, 'style': ['--' for _ in range(len(y_line_normed))]})
    # ax_line = sns.lineplot(data=data, x='x', y='y', ax=ax)
    # ax_line.lines[0].set_linestyle(linestyle)

    sns.lineplot(data=data, x='x', y='y', style=True, dashes=[(2, 2)], color=color, linewidth=2)

    return


# vz_over_vth = 0.8
# vz_over_vth = 0.56
vz_over_vth = 1.025  # mean of vz in loss cone
m_curr = 2
offset = 2.5 / m_curr
slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
alpha_const_omega_mass2_right = offset + slope * beta_loop_list
alpha_const_omega_mass2_left = offset - slope * beta_loop_list
m_curr = 3
offset = 2.5 / m_curr
slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
alpha_const_omega_mass3_right = offset + slope * beta_loop_list
alpha_const_omega_mass3_left = offset - slope * beta_loop_list

### PLOTS

annot = False
annot_fontsize = 8
annot_fmt = '.2f'

# yticklabels = alpha_loop_list
# y_label = '$f_{\\omega}$'

_, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
q = Z_ion * settings['e']  # Coulomb
omega0 = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)

_, _, mi, _, Z_ion = define_plasma_parameters(gas_name='deuterium')
q = Z_ion * settings['e']  # Coulomb
omega0_D = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)

omega = alpha_loop_list * field_dict['omega_cyclotron']
yticklabels = ['{:.2f}'.format(w) for w in omega / omega0]
y_array = alpha_loop_list * field_dict['omega_cyclotron'] / omega0
y_label = '$\\omega / \\omega_{0,T}$'

x_array = beta_loop_list
x_label = '$k/\\left( 2 \\pi m^{-1} \\right)$'

X, Y = np.meshgrid(x_array, y_array)

for alpha, beta, set_name in zip(select_alpha_list, select_beta_list, set_name_list):
    ind_alpha = np.where(alpha_loop_list >= alpha)[0][0]
    ind_beta = np.where(beta_loop_list >= beta)[0][0]

    ## Print for python mm_rate_eqs
    # print('set', set_name, 'alpha=', alpha, 'omega/omega0=', alpha * field_dict['omega_cyclotron'] / omega0, 'beta=', beta)
    # print('set', set_name, ', alpha=', alpha,
    #       ', omega/omega0=', '{:.3f}'.format(alpha * field_dict['omega_cyclotron'] / omega0),
    #       ', beta=', beta, ', gas=', gas_name, ', induced_fields_factor=', 1)
    # print('RF_rc_list += [' + '{:.3f}'.format(N_rc_1[ind_beta, ind_alpha]) + ']')
    # print('RF_lc_list += [' + '{:.3f}'.format(N_lc_1[ind_beta, ind_alpha]) + ']')
    # print('RF_cr_list += [' + '{:.3f}'.format(N_cr_1[ind_beta, ind_alpha]) + ']')
    # print('RF_cl_list += [' + '{:.3f}'.format(N_cl_1[ind_beta, ind_alpha]) + ']')
    # print('RF_rl_list += [' + '{:.3f}'.format(N_rl_1[ind_beta, ind_alpha]) + ']')
    # print('RF_lr_list += [' + '{:.3f}'.format(N_lr_1[ind_beta, ind_alpha]) + ']')
    # print('set', set_name, ', alpha=', alpha,
    #       ', omega/omega0=', '{:.3f}'.format(alpha * field_dict['omega_cyclotron'] / omega0),
    #       ', beta=', beta, ', gas=', gas_name, ', induced_fields_factor=', 0)
    # print('RF_rc_list += [' + '{:.3f}'.format(N_rc_2[ind_beta, ind_alpha]) + ']')
    # print('RF_lc_list += [' + '{:.3f}'.format(N_lc_2[ind_beta, ind_alpha]) + ']')
    # print('RF_cr_list += [' + '{:.3f}'.format(N_cr_2[ind_beta, ind_alpha]) + ']')
    # print('RF_cl_list += [' + '{:.3f}'.format(N_cl_2[ind_beta, ind_alpha]) + ']')
    # print('RF_rl_list += [' + '{:.3f}'.format(N_rl_2[ind_beta, ind_alpha]) + ']')
    # print('RF_lr_list += [' + '{:.3f}'.format(N_lr_2[ind_beta, ind_alpha]) + ']')

    # print('ratio between 1 and 2 values:')
    # print('RF_rc_list += [' + '{:.3f}'.format(N_rc_2[ind_beta, ind_alpha] / N_rc_1[ind_beta, ind_alpha]) + ']')
    # print('RF_lc_list += [' + '{:.3f}'.format(N_lc_2[ind_beta, ind_alpha] / N_lc_1[ind_beta, ind_alpha]) + ']')
    # print('RF_cr_list += [' + '{:.3f}'.format(N_cr_2[ind_beta, ind_alpha] / N_cr_1[ind_beta, ind_alpha]) + ']')
    # print('RF_cl_list += [' + '{:.3f}'.format(N_cl_2[ind_beta, ind_alpha] / N_cl_1[ind_beta, ind_alpha]) + ']')
    # print('RF_rl_list += [' + '{:.3f}'.format(N_rl_2[ind_beta, ind_alpha] / N_rl_1[ind_beta, ind_alpha]) + ']')
    # print('RF_lr_list += [' + '{:.3f}'.format(N_lr_2[ind_beta, ind_alpha] / N_lr_1[ind_beta, ind_alpha]) + ']')

    # shorter notation
    induced_fields_factor = 1
    print('#set', set_name, ', alpha=', alpha,
          ', omega/omega0=', '{:.3f}'.format(alpha * field_dict['omega_cyclotron'] / omega0),
          ', beta=', beta, ', gas=', gas_name, ', induced_fields_factor=', induced_fields_factor)
    print('set_name_list += ["' + set_name + ' (' + gas_name_short + ', iff=' + str(induced_fields_factor) + ')"]')
    print('gas_type_list += ["' + gas_name + '"]')
    RF_str = 'RF_rate_list += [['
    for N_rate in [N_rc_1, N_lc_1, N_cr_1, N_cl_1, N_rl_1, N_lr_1]:
        RF_str += '{:.3f}'.format(N_rate[ind_beta, ind_alpha]) + ', '
    RF_str += ']]'
    print(RF_str)

    induced_fields_factor = 0
    print('#set', set_name, ', alpha=', alpha,
          ', omega/omega0=', '{:.3f}'.format(alpha * field_dict['omega_cyclotron'] / omega0),
          ', beta=', beta, ', gas=', gas_name, ', induced_fields_factor=', induced_fields_factor)
    print('set_name_list += ["' + set_name + ' (' + gas_name_short + ', iff=' + str(induced_fields_factor) + ')"]')
    print('gas_type_list += ["' + gas_name + '"]')
    RF_str = 'RF_rate_list += [['
    for N_rate in [N_rc_2, N_lc_2, N_cr_2, N_cl_2, N_rl_2, N_lr_2]:
        RF_str += '{:.3f}'.format(N_rate[ind_beta, ind_alpha]) + ', '
    RF_str += ']]'
    print(RF_str)

    if plot_rate_eqs_flux:
        flux_str = 'flux = ' + '{:.3f}'.format(rate_eqs_flux_1[ind_beta, ind_alpha])
        print(flux_str)

# ## Print for latex
# print(set_name, '(D) & ',
#       '{:.2f}'.format(alpha * field_dict['omega_cyclotron'] / omega0_D), ' & ',
#       '{:.2f}'.format(alpha * field_dict['omega_cyclotron'] / omega0), ' & ',
#       str(beta), ' & ',
#       '{:.2f}'.format(N_rc_1[ind_beta, ind_alpha]), ' & ',
#       '{:.2f}'.format(N_lc_1[ind_beta, ind_alpha]), ' & ',
#       '{:.2f}'.format(N_cr_1[ind_beta, ind_alpha]), ' & ',
#       '{:.2f}'.format(N_cl_1[ind_beta, ind_alpha]), ' & ',
#       '{:.1f}'.format(N_rc_1[ind_beta, ind_alpha] / N_lc_1[ind_beta, ind_alpha]), "\\\\")
# print(set_name, '(T) & & & &',
#       '{:.2f}'.format(N_rc_2[ind_beta, ind_alpha]), ' & ',
#       '{:.2f}'.format(N_lc_2[ind_beta, ind_alpha]), ' & ',
#       '{:.2f}'.format(N_cr_2[ind_beta, ind_alpha]), ' & ',
#       '{:.2f}'.format(N_cl_2[ind_beta, ind_alpha]), ' & ',
#       '{:.1f}'.format(N_rc_2[ind_beta, ind_alpha] / N_lc_2[ind_beta, ind_alpha]), "\\\\")
# print('\\hline')


def plot_resonance_lines():
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_mass2_right, color='lawngreen')
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_mass2_left, color='lawngreen')
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_mass3_right, color='cyan')
    plot_line_on_heatmap(beta_loop_list, alpha_loop_list, alpha_const_omega_mass3_left, color='cyan')
    return


def plot_interest_points(ax):
    pass
    # for ind_set, (alpha, beta, set_name) in enumerate(zip(select_alpha_list, select_beta_list, set_name_list)):
    #     ind_alpha = np.where(alpha_loop_list >= alpha)[0][0]
    #     ind_beta = np.where(beta_loop_list >= beta)[0][0]
    #     # data = {'y': [ind_alpha], 'x': [ind_beta]}
    #     # print(data)
    #     # sns.scatterplot(data=data, x='x', y='y', size=100, color="b", marker="o")
    #     # sns.scatterplot(data=data, x='x', y='y',
    #     #                 # markersize=2000, color="none", marker="o",
    #     #                 # edgecolor='b'
    #     #                 kwargs={'s': 50}
    #     #                 )
    #     # sns.scatterplot(data=data, x='x', y='y', marker='o',ms=60,mec='r',mfc='none')
    #     # points_x_loc = ind_beta + 0.5
    #     # points_y_loc = ind_alpha + 0.5
    #     points_x_loc = beta
    #     points_y_loc = alpha * field_dict['omega_cyclotron'] / omega0
    #     ax.scatter(points_x_loc, points_y_loc,
    #                s=200, marker='o',
    #                # alpha=0.5,
    #                # facecolor='none',
    #                facecolor='b',
    #                edgecolor='b', linewidth=2)
    #     # points_x_loc += 0.5
    #     # points_y_loc += 0.5
    #     # points_x_loc -= 0.04
    #     # points_y_loc -= 0.02
    #     if len(set_name) == 1:
    #         points_x_loc -= 0.05
    #         points_y_loc -= 0.03
    #     else:
    #         points_x_loc -= 0.10
    #         points_y_loc -= 0.03
    #     ax.text(points_x_loc, points_y_loc,
    #             set_name, color='w',
    #             fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 12}, )
    #     # ax.text(ind_beta - 0.125 + 0.5, ind_alpha - 0.125 + 0.5, set_name, color='w',
    #     #         fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 12}, )


def plot_colormesh(Z, title, fig=None, ax=None, vmin=None, vmax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vmin is None:
        vmin = np.nanmin(Z)
    if vmax is None:
        vmax = np.nanmax(Z)

    # plt.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
    # plt.xlabel(x_label, fontsize=axes_label_size)
    # plt.ylabel(y_label, fontsize=axes_label_size)
    # plt.title(title, fontsize=title_fontsize)
    # plt.colorbar()
    # plt.tight_layout()

    c = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(x_label, fontsize=axes_label_size)
    ax.set_ylabel(y_label, fontsize=axes_label_size)
    ax.set_title(title, fontsize=title_fontsize)
    fig.colorbar(c, ax=ax)
    fig.set_layout_engine(layout='tight')
    update_format_coord(X, Y, Z, ax=ax)
    return ax


# lower resolution for xticks and yticks
xticklabels_str = []
yticklabels_str = []
for i in range(len(beta_loop_list)):
    if np.mod(i, 2) == 0:
        # xticklabels_str += [str(int(beta_loop_list[i]))]
        yticklabels_str += [str(yticklabels[i])]
        xticklabels_str += [str(beta_loop_list[i])]
        # yticklabels_str += [str(alpha_loop_list[i])]
    else:
        xticklabels_str += ['']
        yticklabels_str += ['']
# beta_loop_list = beta_loop_list
# yticklabels = yticklabels


# do_plots = False
do_plots = True

if do_plots == True:

    #########################
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))

    Z = selectivity_1
    title = '$\\bar{N}_{rc} / \\bar{N}_{lc}$ (' + label_1 + ')'
    ax = axes[0, 0]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    plot_interest_points(ax)

    Z = selectivity_2
    title = '$\\bar{N}_{rc} / \\bar{N}_{lc}$ (' + label_2 + ')'
    ax = axes[0, 1]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    plot_interest_points(ax)

    Z = cone_escape_rate_1
    title = '$(N_{rc}-N_{cr})/N_{cone}$ (' + label_1 + ')'
    ax = axes[1, 0]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    # plot_interest_points(ax)

    Z = cone_escape_rate_2
    title = '$(N_{rc}-N_{cr})/N_{cone}$ (' + label_2 + ')'
    ax = axes[1, 1]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    # plot_interest_points(ax)

    #########################
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))

    Z = percent_ok_1
    title = '%ok (' + label_1 + ')'
    ax = axes[0, 0]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    # plot_interest_points(ax)

    Z = E_ratio_mean_1
    title = 'E_ratio_mean (' + label_1 + ')'
    ax = axes[1, 0]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    # plot_interest_points(ax)

    Z = percent_ok_2
    title = '%ok (' + label_2 + ')'
    ax = axes[0, 1]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    plot_interest_points(ax)

    Z = E_ratio_mean_2
    title = 'E_ratio_mean (' + label_2 + ')'
    ax = axes[1, 1]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    plot_interest_points(ax)

    #########################

    processes = ['rc', 'cr', 'rl', 'lc', 'cl', 'lr']
    ind_rows = [0, 0, 0, 1, 1, 1]
    ind_cols = [0, 1, 2, 0, 1, 2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    mat_dict = mat_dict_1
    title_suffix = ' (' + label_1 + ')'
    vmin_list = [0, 0, 0, 0, 0, 0]
    vmax_list = [0.9, 0.05, 0.15, 0.9, 0.05, 0.15]
    # vmax_list = [0.95, 0.05, 0.1, 0.95, 0.05, 0.1]
    # for process, ind_row, ind_col in zip(processes, ind_rows, ind_cols):
    for process, ind_row, ind_col, vmin, vmax in zip(processes, ind_rows, ind_cols, vmin_list, vmax_list):
        Z = mat_dict['N_' + process]
        title = '$N_{' + process + '}$' + title_suffix
        ax = axes[ind_row, ind_col]
        ax = plot_colormesh(Z.T, title, fig=fig, ax=ax,
                            # vmin=None, vmax=None,
                            vmin=vmin, vmax=vmax,
                            )
        plot_interest_points(ax)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    mat_dict = mat_dict_2
    title_suffix = ' (' + label_2 + ')'
    # vmin_list = [0, 0, 0, 0, 0, 0]
    # vmax_list = [0.9, 0.15, 0.15, 0.9, 0.15, 0.15]
    # vmax_list = [0.95, 0.1, 0.1, 0.95, 0.1, 0.1]
    # for process, ind_row, ind_col in zip(processes, ind_rows, ind_cols):
    for process, ind_row, ind_col, vmin, vmax in zip(processes, ind_rows, ind_cols, vmin_list, vmax_list):
        Z = mat_dict['N_' + process]
        title = '$N_{' + process + '}$' + title_suffix
        ax = axes[ind_row, ind_col]
        ax = plot_colormesh(Z.T, title, fig=fig, ax=ax,
                            # vmin=None, vmax=None,
                            vmin=vmin, vmax=vmax,
                            )
        plot_interest_points(ax)

    # fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    # mat_dict = mat_dict_2
    # gas_name = ' (diff)'
    # for process, ind_row, ind_col in zip(processes, ind_rows, ind_cols):
    #     Z = mat_dict_2['N_' + process] - mat_dict_1['N_' + process]
    #     title = '$N_{' + process + '}$' + gas_name
    #     ax = axes[ind_row, ind_col]
    #     ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)

    #########################
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    Z = np.log(rate_eqs_flux_1)
    title = 'flux (' + label_1 + ')'
    ax = axes[0]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    # plot_interest_points(ax)

    Z = np.log(rate_eqs_flux_2)
    title = 'flux (' + label_2 + ')'
    ax = axes[1]
    ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # plot_resonance_lines()
    # plot_interest_points(ax)
