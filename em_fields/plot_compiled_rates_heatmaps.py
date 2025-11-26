import pickle

import matplotlib.pyplot as plt
import numpy as np
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
# cmap = 'turbo'
cmap = 'coolwarm'
figsize = (6, 6)

axes_label_size = 14
title_fontsize = 16

# plt.close('all')

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
# save_dir += '/set54_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set55_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set56_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set57_B0_1T_l_1m_Post_Rm_5_r0max_30cm_intervals_D_T/'
# save_dir += '/set58_B0_1T_l_1m_Post_Rm_10_r0max_30cm_intervals_D_T/'
save_dir += '/set59_B0_1T_l_1m_Post_Rm_5_r0max_30cm/'


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
# # # E_RF_kVm = 25  # [kV/m]
E_RF_kVm = 50  # [kV/m]
RF_type = 'magnetic_transverse'
# B_RF = 0.02  # [T]
B_RF = 0.04  # [T]

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'

# with_kr_correction = False
with_kr_correction = True

induced_fields_factor = 1
# induced_fields_factor = 0.5
# induced_fields_factor = 0.1
# induced_fields_factor = 0.01
# induced_fields_factor = 0
time_step_tau_cyclotron_divisions = 50
# time_step_tau_cyclotron_divisions = 100
sigma_r0 = 0
# sigma_r0 = 0.05
# sigma_r0 = 0.1
sigma_r0 = 0.3
radial_distribution = 'uniform'

# theta_type = 'sign_vz0'
theta_type = 'sign_vz'

loss_cone_condition = 'B_total'  # correct form
# loss_cone_condition = 'B_axial'  # testing the incorrect way implemented in the past
# loss_cone_condition = 'old_compilation'

plot_D = False
# plot_D = True
plot_T = True

load_smoothed_rates = False
# load_smoothed_rates = True

if plot_D:
    gas_name = 'deuterium'
    # gas_name = 'DT_mix'
    # gas_name = 'tritium'

    set_name = 'compiled_'
    if load_smoothed_rates:
        set_name = 'smooth_compiled_'
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
    if loss_cone_condition == 'B_axial':
        set_name += '_LCcondBz'
    if loss_cone_condition == 'old_compilation':
        set_name += '_LCcondOLD'

    title1 = set_name
    print(set_name)
    save_file = save_dir + '/' + set_name + '.mat'

    mat_dict_1 = loadmat(save_file)
    alpha_loop_list = mat_dict_1['alpha_loop_list'][0]
    beta_loop_list = mat_dict_1['beta_loop_list'][0]
    N_rc_1 = mat_dict_1['N_rc_end']
    N_lc_1 = mat_dict_1['N_lc_end']
    N_cr_1 = mat_dict_1['N_cr_end']
    N_cl_1 = mat_dict_1['N_cl_end']
    N_rl_1 = mat_dict_1['N_rl_end']
    N_lr_1 = mat_dict_1['N_lr_end']
    percent_ok_1 = mat_dict_1['percent_ok']
    E_ratio_1 = mat_dict_1['E_ratio']
    selectivity_LC_1 = N_rc_1 / N_lc_1
    selectivity_trapped_1 = N_cr_1 / N_cr_1
    cone_escape_rate_1 = (N_rc_1 * LC_ini_fraction - N_cr_1 * trapped_ini_fraction) / LC_ini_fraction

if plot_T:
    # gas_name = 'deuterium'
    # gas_name = 'DT_mix'
    gas_name = 'tritium'
    set_name = 'compiled_'
    if load_smoothed_rates:
        set_name = 'smooth_compiled_'
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
    if loss_cone_condition == 'B_axial':
        set_name += '_LCcondBz'
    if loss_cone_condition == 'old_compilation':
        set_name += '_LCcondOLD'


    title2 = set_name
    print(set_name)
    save_file = save_dir + '/' + set_name + '.mat'

    mat_dict_2 = loadmat(save_file)
    N_rc_2 = mat_dict_2['N_rc_end']
    N_lc_2 = mat_dict_2['N_lc_end']
    N_cr_2 = mat_dict_2['N_cr_end']
    N_cl_2 = mat_dict_2['N_cl_end']
    N_rl_2 = mat_dict_2['N_rl_end']
    N_lr_2 = mat_dict_2['N_lr_end']
    percent_ok_2 = mat_dict_2['percent_ok']
    E_ratio_2 = mat_dict_2['E_ratio']
    selectivity_LC_2 = N_rc_2 / N_lc_2
    selectivity_trapped_2 = N_cr_2 / N_cl_2
    cone_escape_rate_2 = (N_rc_2 * LC_ini_fraction - N_cr_2 * trapped_ini_fraction) / LC_ini_fraction
    alpha_loop_list = mat_dict_2['alpha_loop_list'][0]
    beta_loop_list = mat_dict_2['beta_loop_list'][0]
### PLOTS

annot = False
annot_fontsize = 8
annot_fmt = '.2f'

_, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
q = Z_ion * settings['e']  # Coulomb
omega0 = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)

omega = alpha_loop_list * field_dict['omega_cyclotron']
yticks = np.arange(0.6, 2.0, 0.2)
y_array = alpha_loop_list * field_dict['omega_cyclotron'] / omega0
y_label = '$\\omega / \\omega_{0,T}$'

x_array = beta_loop_list
x_label = '$k/\\left( 2 \\pi m^{-1} \\right)$'

X, Y = np.meshgrid(x_array, y_array)

# definitions for theoretic resonance lines
# vz_over_vth = 1.025  # mean of vz in loss cone
vz_over_vth = np.pi ** (-0.5) * (1 + np.sqrt(1 - 1 / field_dict['Rm']))  # mean of vz in loss cone
# m_curr = 2
# offset = 2.5 / m_curr
# slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
offset = 3 / 2
slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
alpha_const_omega_mass2_right = offset + slope * beta_loop_list
alpha_const_omega_mass2_left = offset - slope * beta_loop_list
m_curr = 3
# offset = 2.5 / m_curr
# slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
offset = 1
slope = 2 * np.pi * vz_over_vth * settings['v_th'] / field_dict['omega_cyclotron']
alpha_const_omega_mass3_right = offset + slope * beta_loop_list
alpha_const_omega_mass3_left = offset - slope * beta_loop_list


def plot_line_on_heatmap(ax, x_heatmap, y_line, color='k'):
    ax.plot(x_heatmap, y_line, color=color, linestyle='--', linewidth=2)
    return


def plot_resonance_lines(ax, gas_name='D'):
    if gas_name == 'D':
        plot_line_on_heatmap(ax, beta_loop_list, alpha_const_omega_mass2_right)
        plot_line_on_heatmap(ax, beta_loop_list, alpha_const_omega_mass2_left)
    else:
        plot_line_on_heatmap(ax, beta_loop_list, alpha_const_omega_mass3_right)
        plot_line_on_heatmap(ax, beta_loop_list, alpha_const_omega_mass3_left)
    return

def plot_colormesh(Z, title, fig=None, ax=None, vmin=None, vmax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vmin is None:
        vmin = np.nanmin(Z)
    if vmax is None:
        vmax = np.nanmax(Z)

    c = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_yticks(yticks)
    ax.set_xlabel(x_label, fontsize=axes_label_size)
    ax.set_ylabel(y_label, fontsize=axes_label_size)
    ax.set_title(title, fontsize=title_fontsize)
    fig.colorbar(c, ax=ax)
    fig.set_layout_engine(layout='tight')
    update_format_coord(X, Y, Z, ax=ax)
    return ax


# do_plots = False
do_plots = True

if do_plots == True:

    # #########################
    # fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    #
    # Z = selectivity_1
    # title = '$\\bar{N}_{rc} / \\bar{N}_{lc}$ (D)'
    # ax = axes[0, 0]
    # ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # # plot_resonance_lines()
    # plot_interest_points(ax)
    #
    # Z = selectivity_2
    # title = '$\\bar{N}_{rc} / \\bar{N}_{lc}$ (T)'
    # ax = axes[0, 1]
    # ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # # plot_resonance_lines()
    # plot_interest_points(ax)

    # Z = cone_escape_rate_1
    # title = '$(N_{rc}-N_{cr})/N_{cone}$ (D)'
    # ax = axes[1, 0]
    # ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # # plot_resonance_lines()
    # # plot_interest_points(ax)
    #
    # Z = cone_escape_rate_2
    # title = '$(N_{rc}-N_{cr})/N_{cone}$ (T)'
    # ax = axes[1, 1]
    # ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    # # plot_resonance_lines()
    # # plot_interest_points(ax)

    #########################

    processes = ['rc', 'cr', 'rl', 'lc', 'cl', 'lr']
    ind_rows = [0, 0, 0, 1, 1, 1]
    ind_cols = [0, 1, 2, 0, 1, 2]
    process_colors = ['b', 'r', 'k', 'g', 'orange', 'brown']
    # vmin_list = [None for _ in range(len(processes))]
    # vmax_list = [None for _ in range(len(processes))]
    vmin_list = [0.2, 0, 0, 0.2, 0, 0]
    vmax_list = [0.8, 0.08, 0.12, 0.8, 0.08, 0.12]

    # rate values
    if plot_D:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        mat_dict = mat_dict_1
        gas_name = 'D'
        for process, ind_row, ind_col, vmin, vmax in zip(processes, ind_rows, ind_cols, vmin_list, vmax_list):
            Z = mat_dict['N_' + process + '_end']
            title = '$N_{' + process + '}$ (' + gas_name + ')'
            ax = axes[ind_row, ind_col]
            ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=vmin, vmax=vmax)
            plot_resonance_lines(ax, gas_name=gas_name)
        fig.suptitle(title1, fontsize=title_fontsize)

    if plot_T:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        mat_dict = mat_dict_2
        gas_name = 'T'
        for process, ind_row, ind_col, vmin, vmax in zip(processes, ind_rows, ind_cols, vmin_list, vmax_list):
            Z = mat_dict['N_' + process + '_end']
            title = '$N_{' + process + '}$ (' + gas_name + ')'
            ax = axes[ind_row, ind_col]
            ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=vmin, vmax=vmax)
            plot_resonance_lines(ax, gas_name=gas_name)
        fig.suptitle(title2, fontsize=title_fontsize)

    #
    # # rate values after smoothing
    # smoothing_sigma = 1
    #
    # fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    # mat_dict = mat_dict_1
    # gas_name = ' (D) smoothed'
    # for process, ind_row, ind_col in zip(processes, ind_rows, ind_cols):
    #     Z = mat_dict['N_' + process + '_end']
    #     mat_dict['N_' + process + '_end']
    #     smoothed_image = gaussian_filter(Z, sigma=smoothing_sigma)
    #     title = '$N_{' + process + '}$' + gas_name
    #     ax = axes[ind_row, ind_col]
    #     ax = plot_colormesh(smoothed_image.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #
    # fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    # mat_dict = mat_dict_2
    # gas_name = ' (T) smoothed'
    # for process, ind_row, ind_col in zip(processes, ind_rows, ind_cols):
    #     Z = mat_dict['N_' + process + '_end']
    #     smoothed_image = gaussian_filter(Z, sigma=smoothing_sigma)
    #     title = '$N_{' + process + '}$' + gas_name
    #     ax = axes[ind_row, ind_col]
    #     ax = plot_colormesh(smoothed_image.T, title, fig=fig, ax=ax, vmin=None, vmax=None)

    #########################

    mat_dict_list, ind_row_list, gas_name_list = [], [], []
    if plot_D:
        mat_dict_list += [mat_dict_1]
        ind_row_list += [0]
        gas_name_list += ['D']
    if plot_T:
        mat_dict_list += [mat_dict_2]
        ind_row_list += [1]
        gas_name_list += ['T']

    # fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    # fig.suptitle(title2, fontsize=title_fontsize)
    #
    # for mat_dict, ind_row, gas_name in zip([mat_dict_1, mat_dict_2], [0, 1], ['D', 'T']):
    #     E_ratio = mat_dict['E_ratio']
    #     selectivity_LC = mat_dict['N_rc_end'] / mat_dict['N_lc_end']
    #     selectivity_trapped = mat_dict['N_cr_end'] / mat_dict['N_cl_end']
    #
    #     Z = selectivity_LC
    #     title = '$N_{rc} / N_{lc}$ (' + gas_name + ')'
    #     ax = axes[ind_row, 0]
    #     ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #     plot_resonance_lines(ax, gas_name=gas_name)
    #
    #     Z = selectivity_trapped
    #     title = '$N_{cr} / N_{cl}$ (' + gas_name + ')'
    #     ax = axes[ind_row, 1]
    #     ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #     plot_resonance_lines(ax, gas_name=gas_name)
    #
    #     Z = E_ratio
    #     title = '$\\bar{E}_{fin}/\\bar{E}_{ini}$ (' + gas_name + ')'
    #     ax = axes[ind_row, 2]
    #     ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #     plot_resonance_lines(ax, gas_name=gas_name)
    #
    #     # # estimate the power per particle, not just energy difference
    #     # E_ini_per_particle = settings['kB_eV'] * settings['T_keV'] * 1e3 # [Joule]
    #     # N_particles = 1e21 # density 1e21[m^-3] in volume 1[m^3]
    #     # E_ini_total = E_ini_per_particle * N_particles  # [Joule]
    #     # E_fin_total = E_ini_total * E_ratio
    #     # power_total_W = (E_fin_total - E_ini_total) / settings['t_max'] # [Watt=Joule/s]
    #     # power_total_MW = power_total_W / 1e6
    #     # Z = power_total_MW
    #     # title = 'Power [MW] (' + gas_name + ')'
    #     # ax = axes[ind_row, 2]
    #     # ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #     # plot_resonance_lines(ax, gas_name=gas_name)

    # fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    # fig.suptitle(title2, fontsize=title_fontsize)
    #
    # for mat_dict, ind_row, gas_name in zip(mat_dict_list, ind_row_list, gas_name_list):
    #     Z = mat_dict['E_ratio_R']
    #     title = 'E_ratio_R (' + gas_name + ')'
    #     ax = axes[ind_row, 0]
    #     ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #     plot_resonance_lines(ax, gas_name=gas_name)
    #
    #     # Z = mat_dict['E_ratio_C']
    #     Z = mat_dict['E_ratio_L']  # C,L were mixed in compilation
    #     title = 'E_ratio_C (' + gas_name + ')'
    #     ax = axes[ind_row, 1]
    #     ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #     plot_resonance_lines(ax, gas_name=gas_name)
    #
    #     # Z = mat_dict['E_ratio_L']
    #     Z = mat_dict['E_ratio_C']  # C,L were mixed in compilation
    #     title = 'E_ratio_L (' + gas_name + ')'
    #     ax = axes[ind_row, 2]
    #     ax = plot_colormesh(Z.T, title, fig=fig, ax=ax, vmin=None, vmax=None)
    #     plot_resonance_lines(ax, gas_name=gas_name)



# # estimate power particle gets in constant electric field
# a_constE = settings['q'] * 50e3 / settings['mi']
# dt = 1.25e-6 # [s]
# denergy_constE = 0.5 * settings['mi'] * (a_constE * dt) ** 2
# # power_constE = denergy_constE / dt # [Watt]
#
# # estimate for a change of order of the temperature as in the RF
# denergy_RF = settings['kB_eV'] * settings['T_keV'] * 1e3


### saving figures
# fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2025/pics/'
#
# file_name = 'compiled_rates'
# if 'smooth' in set_name:
#     file_name += '_smooth'
# if RF_type == 'electric_transverse': file_name += '_REF'
# else: file_name += '_RMF'
# if induced_fields_factor < 1.0: file_name += '_iff' + str(induced_fields_factor)
#
# plt.figure(1)
# plt.savefig(fig_save_dir + file_name + '_D' + '.pdf', format='pdf', dpi=600)
#
# plt.figure(2)
# plt.savefig(fig_save_dir + file_name + '_T' + '.pdf', format='pdf', dpi=600)
