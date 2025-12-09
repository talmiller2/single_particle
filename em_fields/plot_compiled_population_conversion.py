import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_cyclotron_angular_frequency

# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 18})
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
# save_dir += '/set54_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set55_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set56_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set57_B0_1T_l_1m_Post_Rm_5_r0max_30cm_intervals_D_T/'
# save_dir += '/set58_B0_1T_l_1m_Post_Rm_10_r0max_30cm_intervals_D_T/'
# save_dir += '/set59_B0_1T_l_1m_Post_Rm_5_r0max_30cm/'
# save_dir += '/set60_B0_1T_l_1m_Post_Rm_5_r0max_30cm_tmax_10tau/'  # for longer tmax testing
save_dir += '/set61_B0_1T_l_1m_Post_Rm_5_r0max_10cm/'


save_dir_curr = save_dir + 'without_RF'
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

LC_ini_fraction = np.sin(np.arcsin(field_dict['Rm'] ** (-0.5)) / 2) ** 2
trapped_ini_fraction = 1 - 2 * LC_ini_fraction

normalize_curves = False
# normalize_curves = True

RF_type = 'electric_transverse'
# E_RF_kVm = 25  # [kV/m]
E_RF_kVm = 50  # [kV/m]
RF_type = 'magnetic_transverse'
# # B_RF = 0.02  # [T]
# B_RF = 0.04  # [T]
B_RF = 0.05  # [T]

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
# sigma_r0 = 0
# sigma_r0 = 0.05
sigma_r0 = 0.1
# sigma_r0 = 0.3
radial_distribution = 'uniform'

# theta_type = 'sign_vz0'
theta_type = 'sign_vz'

loss_cone_condition = 'B_total'  # correct form
# loss_cone_condition = 'B_axial' # testing the incorrect way implemented in the past
# loss_cone_condition = 'old_compilation'  # used the old compilation code

# plot_D = False
plot_D = True
plot_T = True

if plot_D:
    gas_name = 'deuterium'
    # gas_name = 'DT_mix'
    # gas_name = 'tritium'

    set_name = 'compiled_'
    # set_name = 'smooth_compiled_'
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

    title_1 = set_name
    print(set_name)
    save_file = save_dir + '/' + set_name + '.mat'
    mat_dict_1 = loadmat(save_file)
    alpha_loop_list = mat_dict_1['alpha_loop_list'][0]
    beta_loop_list = mat_dict_1['beta_loop_list'][0]

if plot_T:
    # gas_name = 'deuterium'
    # gas_name = 'DT_mix'
    gas_name = 'tritium'
    # set_name = 'smooth_compiled_'
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
    if loss_cone_condition == 'B_axial':
        set_name += '_LCcondBz'
    if loss_cone_condition == 'old_compilation':
        set_name += '_LCcondOLD'

    title_2 = set_name
    print(set_name)
    save_file = save_dir + '/' + set_name + '.mat'
    mat_dict_2 = loadmat(save_file)

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
y_array = alpha_loop_list * field_dict['omega_cyclotron'] / omega0
y_label = '$\\omega / \\omega_{0,T}$'

x_array = beta_loop_list
x_label = '$k/\\left( 2 \\pi m^{-1} \\right)$'

loss_cone_angle_rad = np.arcsin(1 / np.sqrt(field_dict['Rm']))
alpha = omega_loss_cone_fraction = np.sin(loss_cone_angle_rad / 2) ** 2
solid_angle_factor = (1 - 2 * alpha) / alpha

# do_plots = False
do_plots = True

if do_plots == True:

    if normalize_curves:
        processes = ['rc', 'cr', 'lc', 'cl']
        ind_rows = [0, 0, 1, 1]
        ind_cols = [0, 1, 0, 1]
        process_colors = ['b', 'r', 'g', 'orange']
    else:
        processes = ['rc', 'cr', 'rl', 'lc', 'cl', 'lr']
        ind_rows = [0, 0, 0, 1, 1, 1]
        ind_cols = [0, 1, 2, 0, 1, 2]
        process_colors = ['b', 'r', 'k', 'g', 'orange', 'brown']

    # processes = ['rc', 'cr']
    # ind_rows = [0, 0]
    # ind_cols = [0, 1]
    # process_colors = ['b', 'r']

    # processes = ['rc', 'lc']
    # ind_rows = [0, 1]
    # ind_cols = [0, 0]
    # process_colors = ['b', 'g']

    if normalize_curves:
        fac_list = [1 if p in ['rc', 'lc'] else solid_angle_factor for p in processes]
    else:
        fac_list = [1 for p in processes]

    # pick a lower alpha, beta resolution for the mega figure
    # inds_alpha = list(range(0, len(alpha_loop_list), 1))  # full res
    # inds_beta = list(range(0, len(beta_loop_list), 1))
    # inds_alpha = list(range(0, len(alpha_loop_list), 2))
    # inds_beta = list(range(0, len(beta_loop_list), 2))
    # inds_alpha = list(range(0, len(alpha_loop_list), 5))
    # inds_beta = list(range(0, len(beta_loop_list), 5))
    # inds_alpha = list(range(0, len(alpha_loop_list), 10))
    # inds_beta = list(range(0, len(beta_loop_list), 10))
    # inds_alpha = [10]
    # inds_beta = [0, 10, 20]
    # inds_alpha = [12]
    # inds_beta = [2, 10, 18]
    # inds_alpha = inds_alpha[::-1]
    # inds_alpha = [0, 2, 3, 4, 6]
    # inds_beta = [0, 2, 3, 4, 6]
    inds_alpha = [15]
    inds_beta = [20]

    # curves

    def plot_2d_matrix_of_population_conversion_plots(mat_dict, title, plot_axes_values=True, display_type='matrix'):
        # fig, axes = plt.subplots(len(inds_alpha), len(inds_beta), figsize=(15, 7))
        fig, axes = plt.subplots(len(inds_alpha), len(inds_beta), figsize=(10, 7))
        # axes = np.atleast_2d(axes)
        # plt.suptitle(title, y=0.92)
        # mat_dict = mat_dict
        t_array = mat_dict['t_array_normed'][0]
        for i_ax, ind_beta in enumerate(inds_beta):
            for j_ax, ind_alpha in enumerate(inds_alpha):

                if len(inds_beta) == 1 and len(inds_alpha) == 1:
                    ax = axes
                elif len(inds_beta) == 1:
                    ax = axes[j_ax]
                elif len(inds_alpha) == 1:
                    ax = axes[i_ax]
                else:
                    # ax = axes[j_ax, i_ax]
                    ax = axes[
                        len(inds_alpha) - 1 - j_ax, i_ax]  # change the direction of rows for alpha to rise in the up direction

                for process, process_color, fac in zip(processes, process_colors, fac_list):
                    curve_mean = fac * mat_dict['N_' + process + '_curve_mean'][ind_beta, ind_alpha]
                    curve_std = fac * mat_dict['N_' + process + '_curve_std'][ind_beta, ind_alpha]
                    ax.plot(t_array, curve_mean, color=process_color)
                    # num_sigmas = 1
                    num_sigmas = 2
                    ax.fill_between(t_array, y1=curve_mean + num_sigmas * curve_std,
                                    y2=curve_mean - num_sigmas * curve_std, color=process_color,
                                    alpha=0.5, label='$N_{' + process + '}$')
                    if display_type == 'matrix':
                        ## for matrix display:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ## for single row display:
                        ax.set_xlabel('$t / \\tau_{th}$')
                        if i_ax == 0 and j_ax == 0:
                            ax.set_ylabel('$\\Delta N / N_0$')
                    ax.set_ylim([0, 1])

                # if j_ax == 0 and i_ax == 0:
                # if j_ax == len(inds_alpha) - 1 and i_ax == 0:
                if j_ax == len(inds_alpha) - 1 and i_ax == len(inds_beta) - 1:
                    if display_type == 'matrix':
                        ax.legend(bbox_to_anchor=(1, 0.5))
                    else:
                        ax.legend()
                        ## for debugging
                        # ax.set_title('$\\alpha=' + str(alpha_loop_list[ind_alpha]) + ', \\beta=' + str(
                        #     beta_loop_list[ind_beta]) + '$',
                        #              # fontsize=8, y=0.8
                        #              )

                    ## for matrix display:
        # plt.subplots_adjust(hspace=0, wspace=0)  # hspace for vertical space, wspace for horizontal space

        # Add a big common x-axis label for beta values (across columns)
        if plot_axes_values:
            fig.text(x=0.51, y=0.04, s=x_label, fontsize=axes_label_size, ha='center', va='center')
            for i_ax, ind_beta in enumerate(inds_beta):
                fig.text(x=0.19 + i_ax * (0.8 / len(inds_beta)), y=0.08,
                         s=f'${beta_loop_list[ind_beta]}$',
                         ha='center', va='center',
                         fontsize=axes_label_size)

            # Add a big common y-axis label for alpha values (across rows)
            fig.text(x=0.07, y=0.5, s=y_label, rotation=90, fontsize=axes_label_size, ha='center', va='center')
            for j_ax, ind_alpha in enumerate(inds_alpha):
                fig.text(
                    # x=0.1,
                    x=0.095,
                    # y=0.8 - j_ax * (0.77 / len(inds_alpha)),
                    y=0.175 + j_ax * (0.8 / len(inds_alpha)),
                         # s=f'${alpha_loop_list[ind_alpha]}$',
                         s=f'${np.round(y_array[ind_alpha], 2)}$',
                         ha='center', va='center',
                         fontsize=axes_label_size)

        # fig.set_layout_engine(layout='tight')
        return fig, axes


    def plot_2d_matrix_of_E_ratio_plots(mat_dict, title, plot_axes_values=True):
        E_ratio_types = ['', '_R', '_L', '_C']
        E_ratio_labels = ['tot', 'R', 'L', 'C']
        colors = ['k', 'b', 'g', 'r']

        fig, axes = plt.subplots(len(inds_alpha), len(inds_beta), figsize=(10, 7))
        t_array = mat_dict['t_array_normed'][0]
        for i_ax, ind_beta in enumerate(inds_beta):
            for j_ax, ind_alpha in enumerate(inds_alpha):

                if len(inds_beta) == 1:
                    ax = axes[j_ax]
                elif len(inds_alpha) == 1:
                    ax = axes[i_ax]
                else:
                    ax = axes[j_ax, i_ax]

                for E_ratio_type, E_ratio_label, color in zip(E_ratio_types, E_ratio_labels, colors):
                    curve = mat_dict['E_ratio' + E_ratio_type + '_curve'][ind_beta, ind_alpha]
                    ax.plot(t_array, curve, color=color, label=E_ratio_label)
                    ## for matrix display:
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    ## for single row display:
                    ax.set_xlabel('$t / \\tau_{th}$')
                    # if i_ax == 0 and j_ax == 0:
                    if i_ax == 0:
                        ax.set_ylabel('$E/E_0$')
                    # ax.set_ylim([0, 1])
                if j_ax == 0 and i_ax == len(inds_beta) - 1:
                    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left')
                ## for debugging
                ax.set_title(
                    '$\\alpha=' + str(alpha_loop_list[ind_alpha]) + ', \\beta=' + str(beta_loop_list[ind_beta]) + '$',
                    fontsize=8, y=0.8)
        ## for matrix display:
        # plt.subplots_adjust(hspace=0, wspace=0)  # hspace for vertical space, wspace for horizontal space

        # # Add a big common x-axis label for beta values (across columns)
        # if plot_axes_values:
        #     fig.text(x=0.51, y=0.04, s=x_label, fontsize=axes_label_size, ha='center', va='center')
        #     for i_ax, ind_beta in enumerate(inds_beta):
        #         fig.text(x=0.19 + i_ax * (0.8 / len(inds_beta)), y=0.08,
        #                  s=f'${beta_loop_list[ind_beta]}$',
        #                  ha='center', va='center',
        #                  fontsize=axes_label_size)
        #
        #     # Add a big common y-axis label for alpha values (across rows)
        #     fig.text(x=0.07, y=0.5, s=y_label, rotation=90, fontsize=axes_label_size, ha='center', va='center')
        #     for j_ax, ind_alpha in enumerate(inds_alpha):
        #         fig.text(x=0.1, y=0.8 - j_ax * (0.77 / len(inds_alpha)),
        #                  # s=f'${alpha_loop_list[ind_alpha]}$',
        #                  s=f'${np.round(y_array[ind_alpha], 2)}$',
        #                  ha='center', va='center',
        #                  fontsize=axes_label_size)

        # fig.set_layout_engine(layout='tight')
        # fig.subplots_adjust(right=0.82)  # manually pull the right edge left a bit
        fig.suptitle(title)
        return fig, axes


    # plot_axes_values = True
    plot_axes_values = False

    # display_type = 'matrix'
    display_type = 'single'

    if plot_D:
        fig_1, axes_1 = plot_2d_matrix_of_population_conversion_plots(mat_dict_1, title_1, plot_axes_values,
                                                                      display_type)
    if plot_T:
        fig_2, axes_2 = plot_2d_matrix_of_population_conversion_plots(mat_dict_2, title_2, plot_axes_values,
                                                                      display_type)
        # plot_2d_matrix_of_E_ratio_plots(mat_dict_2, title_2, plot_axes_values)

## saving figures
fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2025/pics/'

file_name = 'compiled_population_conversion'
if normalize_curves: file_name += '_normalized'
if RF_type == 'electric_transverse':
    file_name += '_REF'
else:
    file_name += '_RMF'
if induced_fields_factor < 1.0: file_name += '_iff' + str(induced_fields_factor)
# fig_1.savefig(fig_save_dir + file_name + '_D' + '.pdf', format='pdf', dpi=600)
# fig_2.savefig(fig_save_dir + file_name + '_T' + '.pdf', format='pdf', dpi=600)
# fig_2.savefig(fig_save_dir + file_name + '_T_single_row' + '.pdf', format='pdf', dpi=600)
fig_2.savefig(fig_save_dir + file_name + '_T_single' + '.pdf', format='pdf', dpi=600)
