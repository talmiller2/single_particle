import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
# save_dir += '/set39_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set40_B0_1T_l_1m_Logan_Rm_3_intervals_D_T/'
# save_dir += '/set41_B0_1T_l_1m_Post_Rm_3_intervals_D_T_ERF_25/'
# save_dir += '/set45_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set46_B0_2T_l_1m_Post_Rm_3_intervals_D_T/'
save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set49_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'

select_alpha_list = []
select_beta_list = []
set_name_list = []

select_alpha_list += [0.64]
select_beta_list += [-1.8]
set_name_list += ['1']

select_alpha_list += [0.7]
select_beta_list += [-0.8]
set_name_list += ['2']

select_alpha_list += [1.06]
select_beta_list += [-1.8]
set_name_list += ['3']

select_alpha_list += [1.12]
select_beta_list += [1.4]
set_name_list += ['4']

select_alpha_list += [0.88]
select_beta_list += [0]
set_name_list += ['5']


save_dir_curr = save_dir + 'without_RF'
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

LC_ini_fraction = np.sin(np.arcsin(field_dict['Rm'] ** (-0.5)) / 2) ** 2
trapped_ini_fraction = 1 - 2 * LC_ini_fraction

RF_type = 'magnetic_transverse'
# B_RF = 0.001  # T
# B_RF = 0.01  # T
# B_RF = 0.02  # T
B_RF = 0.04  # T
# B_RF = 0.08  # T

gas_name = 'deuterium'
# gas_name = 'DT_mix'
# gas_name = 'tritium'

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'

with_RF_xy_corrections = True
# induced_fields_factor = 1
# induced_fields_factor = 0.5
# induced_fields_factor = 0.1
# induced_fields_factor = 0.01
induced_fields_factor = 0
# time_step_tau_cyclotron_divisions = 20
time_step_tau_cyclotron_divisions = 40
# time_step_tau_cyclotron_divisions = 50
# time_step_tau_cyclotron_divisions = 80
# sigma_r0 = 0
sigma_r0 = 0.1

# theta_type = 'sign_vz0'
# theta_type = 'sign_vz'

set_name = 'compiled_'
# set_name += theta_type + '_'
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
print(set_name)
save_file = save_dir + '/' + set_name + '.mat'

mat_dict = loadmat(save_file)
alpha_loop_list = mat_dict['alpha_loop_list'][0]
beta_loop_list = mat_dict['beta_loop_list'][0]
N_rc_1 = mat_dict['N_rc']
N_lc_1 = mat_dict['N_lc']
N_cr_1 = mat_dict['N_cr']
N_cl_1 = mat_dict['N_cl']
# N_rl_1 = mat_dict['N_rl']
# N_lr_1 = mat_dict['N_lr']
# percent_ok_1 = mat_dict['percent_ok']
# E_ratio_mean_1 = mat_dict['E_ratio_mean']

selectivity = N_rc_1 / N_lc_1
selectivity_trapped = N_cr_1 / N_cl_1
# selectivity_1 = copy.deepcopy(selectivity)
selectivity_1 = selectivity
# selectivity_1 = selectivity_trapped
# selectivity_1 = N_rc
# selectivity_1 = N_lc
# selectivity_1 = N_cr
# selectivity_1 = N_cl
cone_escape_rate_1 = (N_rc_1 * LC_ini_fraction - N_cr_1 * trapped_ini_fraction) / LC_ini_fraction

# gas_name = 'deuterium'
# gas_name = 'DT_mix'
gas_name = 'tritium'

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
print(set_name)
save_file = save_dir + '/' + set_name + '.mat'

mat_dict = loadmat(save_file)
N_rc_2 = mat_dict['N_rc']
N_lc_2 = mat_dict['N_lc']
N_cr_2 = mat_dict['N_cr']
N_cl_2 = mat_dict['N_cl']
# selectivity = mat_dict['selectivity']
selectivity = N_rc_2 / N_lc_2
selectivity_trapped = N_cr_2 / N_cl_2

# selectivity_2 = copy.deepcopy(selectivity)
selectivity_2 = selectivity
cone_escape_rate_2 = (N_rc_2 * LC_ini_fraction - N_cr_2 * trapped_ini_fraction) / LC_ini_fraction


# selectivity_2 = selectivity_trapped
# selectivity_2 = N_rc
# selectivity_2 = N_lc
# selectivity_2 = N_cr
# selectivity_2 = N_cl

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
ylabel = '$\\omega / \\omega_{0,T}$'

for alpha, beta, set_name in zip(select_alpha_list, select_beta_list, set_name_list):
    ind_alpha = np.where(alpha_loop_list >= alpha)[0][0]
    ind_beta = np.where(beta_loop_list >= beta)[0][0]

    ## Print for python mm_rate_eqs
    # print('set', set_name, 'alpha=', alpha, 'omega/omega0=', alpha * field_dict['omega_cyclotron'] / omega0, 'beta=', beta)
    print('set', set_name, 'omega/omega0=', '{:.3f}'.format(alpha * field_dict['omega_cyclotron'] / omega0), 'beta=',
          beta)
    # print('  D:  N_rc=',  '{:.3f}'.format(N_rc_1[ind_beta, ind_alpha]), ' N_lc=',  '{:.3f}'.format(N_lc_1[ind_beta, ind_alpha]),
    #       'N_cr=',  '{:.3f}'.format(N_cr_1[ind_beta, ind_alpha]), 'N_cl=',  '{:.3f}'.format(N_cl_1[ind_beta, ind_alpha]),
    #       's=',  '{:.1f}'.format(N_rc_1[ind_beta, ind_alpha] / N_lc_1[ind_beta, ind_alpha]))
    # print('  T:  N_rc=',  '{:.3f}'.format(N_rc_2[ind_beta, ind_alpha]), ' N_lc=',  '{:.3f}'.format(N_lc_2[ind_beta, ind_alpha]),
    #       'N_cr=',  '{:.3f}'.format(N_cr_2[ind_beta, ind_alpha]), 'N_cl=',  '{:.3f}'.format(N_cl_2[ind_beta, ind_alpha]),
    #       's=',  '{:.1f}'.format(N_rc_2[ind_beta, ind_alpha] / N_lc_2[ind_beta, ind_alpha]))
    print('D:')
    print('RF_rc_list += [' + '{:.3f}'.format(N_rc_1[ind_beta, ind_alpha]) + ']')
    print('RF_lc_list += [' + '{:.3f}'.format(N_lc_1[ind_beta, ind_alpha]) + ']')
    print('RF_cr_list += [' + '{:.3f}'.format(N_cr_1[ind_beta, ind_alpha]) + ']')
    print('RF_cl_list += [' + '{:.3f}'.format(N_cl_1[ind_beta, ind_alpha]) + ']')
    print('T:')
    print('RF_rc_list += [' + '{:.3f}'.format(N_rc_2[ind_beta, ind_alpha]) + ']')
    print('RF_lc_list += [' + '{:.3f}'.format(N_lc_2[ind_beta, ind_alpha]) + ']')
    print('RF_cr_list += [' + '{:.3f}'.format(N_cr_2[ind_beta, ind_alpha]) + ']')
    print('RF_cl_list += [' + '{:.3f}'.format(N_cl_2[ind_beta, ind_alpha]) + ']')

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
    # plt.text(0.58, 0.95, 'D resonance', fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 16},
    #          horizontalalignment='right', verticalalignment='top', color='lawngreen', rotation=70,
    #          transform=fig.axes[0].transAxes)
    # plt.text(0.77, 0.95, 'T resonance', fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 16},
    #          horizontalalignment='right', verticalalignment='top', color='cyan', rotation=70,
    #          transform=fig.axes[0].transAxes)
    return


def plot_interest_points(ax):
    for ind_set, (alpha, beta, set_name) in enumerate(zip(select_alpha_list, select_beta_list, set_name_list)):
        ind_alpha = np.where(alpha_loop_list >= alpha)[0][0]
        ind_beta = np.where(beta_loop_list >= beta)[0][0]
        # data = {'y': [ind_alpha], 'x': [ind_beta]}
        # print(data)
        # sns.scatterplot(data=data, x='x', y='y', size=100, color="b", marker="o")
        # sns.scatterplot(data=data, x='x', y='y',
        #                 # markersize=2000, color="none", marker="o",
        #                 # edgecolor='b'
        #                 kwargs={'s': 50}
        #                 )
        # sns.scatterplot(data=data, x='x', y='y', marker='o',ms=60,mec='r',mfc='none')
        ax.scatter(ind_beta + 0.5, ind_alpha + 0.5, s=200, marker='o',
                   # alpha=0.5,
                   # facecolor='none',
                   facecolor='b',
                   edgecolor='b', linewidth=2)
        ax.text(ind_beta - 0.2 + 0.5, ind_alpha - 0.2 + 0.5, set_name, color='w',
                fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 12}, )
        # ax.text(ind_beta - 0.125 + 0.5, ind_alpha - 0.125 + 0.5, set_name, color='w',
        #         fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 12}, )


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

axes_label_size = 14

# do_plots = False
do_plots = True

if do_plots == True:
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # y = N_rc_1
    y = selectivity_1
    # vmin = np.nanmin(y)
    # vmax = np.nanmax(y)
    vmin = 0
    # vmax = 3
    # vmax = 8
    vmax = 10
    # vmax = 14
    sns.heatmap(y.T, xticklabels=xticklabels_str, yticklabels=yticklabels_str,
                vmin=vmin,
                # vmax=vmax,
                annot=annot,
                annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
                ax=ax,
                # alpha=0.5,
                )
    ax.axes.invert_yaxis()
    plot_resonance_lines()
    plot_interest_points(ax)
    ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
    ax.set_ylabel(ylabel, fontsize=axes_label_size)
    ax.set_title('$\\bar{N}_{rc} / \\bar{N}_{lc}$ (D)', fontsize=20)
    # ax.set_title('$\\bar{N}_{cr} / \\bar{N}_{cl}$ (D)', fontsize=20)
    # fig.set_tight_layout(0.5)
    fig.set_layout_engine(layout='tight')
    plt.yticks(rotation=0)
    text = '(a)'
    plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
             horizontalalignment='left', verticalalignment='top', color='w',
             transform=fig.axes[0].transAxes)
    ax.legend().set_visible(False)

    ###############
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    y = selectivity_2
    # y = N_rc_2
    # vmin = np.nanmin(y)
    # vmax = np.nanmax(y)
    vmin = 0
    # vmax = 3
    # vmax = 8
    vmax = 10
    # vmax = 14
    sns.heatmap(y.T, xticklabels=xticklabels_str, yticklabels=yticklabels_str,
                vmin=vmin,
                # vmax=vmax,
                annot=annot,
                annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
                ax=ax,
                # alpha=0.5,
                )
    ax.axes.invert_yaxis()
    plot_resonance_lines()
    plot_interest_points(ax)
    ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
    ax.set_ylabel(ylabel, fontsize=axes_label_size)
    ax.set_title('$\\bar{N}_{rc} / \\bar{N}_{lc}$ (T)', fontsize=20)
    # ax.set_title('$\\bar{N}_{cr} / \\bar{N}_{cl}$ (T)', fontsize=20)
    # fig.set_tight_layout(0.5)
    fig.set_layout_engine(layout='tight')
    plt.yticks(rotation=0)
    text = '(b)'
    plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
             horizontalalignment='left', verticalalignment='top', color='w',
             transform=fig.axes[0].transAxes)
    ax.legend().set_visible(False)

    ###############
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    y = cone_escape_rate_1
    vmin = np.nanmin(y)
    vmax = np.nanmax(y)
    sns.heatmap(y.T, xticklabels=xticklabels_str, yticklabels=yticklabels_str,
                vmin=vmin,
                # vmax=vmax,
                annot=annot,
                annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
                ax=ax,
                # alpha=0.5,
                )
    ax.axes.invert_yaxis()
    plot_resonance_lines()
    plot_interest_points(ax)
    ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
    ax.set_ylabel(ylabel, fontsize=axes_label_size)
    ax.set_title('$(N_{rc}-N_{cr})/N_{cone}$ (D)', fontsize=20)
    # fig.set_tight_layout(0.5)
    fig.set_layout_engine(layout='tight')
    plt.yticks(rotation=0)
    text = '(b)'
    plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
             horizontalalignment='left', verticalalignment='top', color='w',
             transform=fig.axes[0].transAxes)
    ax.legend().set_visible(False)

    ###############
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    y = cone_escape_rate_2
    vmin = np.nanmin(y)
    vmax = np.nanmax(y)
    sns.heatmap(y.T, xticklabels=xticklabels_str, yticklabels=yticklabels_str,
                vmin=vmin,
                # vmax=vmax,
                annot=annot,
                annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
                ax=ax,
                # alpha=0.5,
                )
    ax.axes.invert_yaxis()
    plot_resonance_lines()
    plot_interest_points(ax)
    ax.set_xlabel('$k/\\left( 2 \\pi m^{-1} \\right)$', fontsize=axes_label_size)
    ax.set_ylabel(ylabel, fontsize=axes_label_size)
    ax.set_title('$(N_{rc}-N_{cr})/N_{cone}$ (T)', fontsize=20)
    # fig.set_tight_layout(0.5)
    fig.set_layout_engine(layout='tight')
    plt.yticks(rotation=0)
    text = '(b)'
    plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
             horizontalalignment='left', verticalalignment='top', color='w',
             transform=fig.axes[0].transAxes)
    ax.legend().set_visible(False)

    ## contour plot
    # plt.figure()
    # X = beta_loop_list
    # Y = omega / omega0
    # levels = [0.5] + [i for i in range(1, 10)]
    # Z = selectivity_1.T
    # contour = plt.contour(X, Y, Z, levels)
    # plt.clabel(contour, inline=True, fontsize=8)
    # Z = selectivity_2.T
    # contour = plt.contour(X, Y, Z, levels, linestyles='--')
    # plt.clabel(contour, inline=True, fontsize=8)

    ## save plots to file
    save_dir = '../../../Papers/texts/paper2024/pics/'

    # file_name = 's_heatmap_D'
    # if RF_type == 'magnetic_transverse':
    #     file_name = 'BRF_' + file_name
    # beingsaved = plt.figure(1)
    # beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
    #
    # file_name = 's_heatmap_T'
    # if RF_type == 'magnetic_transverse':
    #     file_name = 'BRF_' + file_name
    # beingsaved = plt.figure(2)
    # beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
