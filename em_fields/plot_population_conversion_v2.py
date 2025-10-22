import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_thermal_velocity

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set48_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set49_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set54_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# save_dir += '/set56_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
save_dir += '/set57_B0_1T_l_1m_Post_Rm_5_r0max_30cm_intervals_D_T/'

# RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 25  # kV/m
# E_RF_kVm = 25  # kV/m
# E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

RF_type = 'magnetic_transverse'
# B_RF = 0.01  # T
# B_RF = 0.02  # T
B_RF = 0.04  # T
# B_RF = 0.05  # T
# B_RF = 0.1  # T

gas_name_list = []
gas_name_list += ['deuterium']
# gas_name_list += ['DT_mix']
# gas_name_list += ['tritium']

select_alpha_list = []
select_beta_list = []
set_name_list = []

# select_alpha_list += [1.12]
# select_beta_list += [2.0]
# set_name_list += ['T1']

# select_alpha_list += [0.82]
# select_beta_list += [1.8]
# set_name_list += ['T1']

select_alpha_list += [1.3]
select_beta_list += [0.4]
set_name_list += ['T1']

use_RF = True
# use_RF = False
# with_kr_correction = False
with_kr_correction = True
induced_fields_factor = 1
# induced_fields_factor = 0.5
# induced_fields_factor = 0.1
# induced_fields_factor = 0.01
# induced_fields_factor = 0
# time_step_tau_cyclotron_divisions = 20
# time_step_tau_cyclotron_divisions = 40
time_step_tau_cyclotron_divisions = 50
# time_step_tau_cyclotron_divisions = 80
# sigma_r0 = 0
# sigma_r0 = 0.05
sigma_r0 = 0.3
# sigma_r0 = 0.1
radial_distribution = 'uniform'

fig_num = 0

process_names = ['rc', 'lc', 'cr', 'cl', 'rl', 'lr']
process_index_pairs = [(0, 1), (2, 1), (1, 0), (1, 2), (0, 2), (2, 0)]

for gas_name in gas_name_list:
    for ind_set in range(len(set_name_list)):
        fig_num += 1

        compiled_dict = {}

        alpha = select_alpha_list[ind_set]
        beta = select_beta_list[ind_set]
        RF_set_name = set_name_list[ind_set]
        # RF_set_name = '$\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)

        title = 'set ' + RF_set_name + ': '
        if RF_type == 'electric_transverse':
            title += '$E_{RF}$=' + str(E_RF_kVm) + 'kV/m, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
        elif RF_type == 'magnetic_transverse':
            title += '$B_{RF}$=' + str(B_RF) + 'T, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
        title += ', ' + gas_name
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
            if induced_fields_factor < 1.0:
                set_name += '_iff' + str(induced_fields_factor)
            if with_kr_correction == True:
                set_name += '_withkrcor'
        set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
        if sigma_r0 > 0:
            set_name += '_sigmar' + str(sigma_r0)
            if radial_distribution == 'normal':
                set_name += 'norm'
            elif radial_distribution == 'uniform':
                set_name += 'unif'
        set_name += '_' + gas_name
        print(set_name)

        save_dir_curr = save_dir + set_name

        # load runs data
        data_dict_file = save_dir_curr + '.pickle'
        with open(data_dict_file, 'rb') as fid:
            data_dict = pickle.load(fid)
        # print('data_dict.keys', data_dict.keys())
        settings_file = save_dir + 'settings.pickle'
        with open(settings_file, 'rb') as fid:
            settings = pickle.load(fid)
        field_dict_file = save_dir + 'field_dict.pickle'
        with open(field_dict_file, 'rb') as fid:
            field_dict = pickle.load(fid)

        # filter out the particles that ended prematurely
        len_t_expected = len(data_dict['t'][0])
        print(f'len_t_expected={len_t_expected}')
        num_particles = len(data_dict['t'])
        print(f'num_particles={num_particles}')

        t_array = np.array(data_dict['t'][0])
        compiled_dict['t_array'] = t_array
        _, _, mi, _, Z_ion = define_plasma_parameters(gas_name=gas_name)
        v_th = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])
        compiled_dict['t_array_normed'] = data_dict['t'][0] / (settings['l'] / v_th)

        # calc percent_ok
        inds_ok = []
        for ind_particle, t in enumerate(data_dict['t']):
            if len(t) == len_t_expected:
                inds_ok += [ind_particle]
        num_particles_ok = len(inds_ok)
        print(f'num_particles_ok={num_particles_ok}')
        compiled_dict['percent_ok'] = len(inds_ok) / num_particles * 100

        # filter data_dict to only the particles that finished the full calc (inds_ok) and transofrm to np.array
        for key in data_dict.keys():
            data_dict[key] = np.array([data_dict[key][i] for i in inds_ok])

        E_ini_mean = np.nanmean(data_dict['v'][:, 0] ** 2)
        E_fin_mean = np.nanmean(data_dict['v'][:, -1] ** 2)
        compiled_dict['E_ratio'] = E_fin_mean / E_ini_mean

        Rm = field_dict['Rm']
        theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(Rm))
        inds_pop = {}
        for pop in ['R', 'L', 'C']:
            inds_pop[pop] = []
        inds_particles_R, inds_particles_L, inds_particles_C = [], [], []
        for i in range(num_particles_ok):
            vt0 = data_dict['v_transverse'][i, 0]
            vz0 = data_dict['v_axial'][i, 0]
            theta0 = np.mod(360 / (2 * np.pi) * np.arctan(vt0 / vz0), 180)
            if theta0 <= theta_LC:
                pop = 'R'
            elif theta0 > theta_LC and theta0 < 180 - theta_LC:
                pop = 'C'
            else:
                pop = 'L'
            inds_pop[pop] += [i]

        for pop in ['R', 'L', 'C']:
            E_ini_mean = np.nanmean(data_dict['v'][inds_pop[pop], 0] ** 2)
            E_fin_mean = np.nanmean(data_dict['v'][inds_pop[pop], -1] ** 2)
            compiled_dict['E_ratio_' + pop] = E_fin_mean / E_ini_mean

        number_of_time_intervals = len(data_dict['t'][0])
        num_bootstrap_samples = 10  # small for testing
        # num_bootstrap_samples = 50
        N_theta = 3
        particles_counter_mat_4d = np.zeros([N_theta, N_theta, number_of_time_intervals, num_bootstrap_samples])

        for ind_boot in range(num_bootstrap_samples):
            print(f'ind_boot: {ind_boot + 1}/{num_bootstrap_samples}')
            if ind_boot == 0:
                inds_particles = range(num_particles_ok)
            else:
                inds_particles = np.random.randint(low=0, high=num_particles_ok,
                                                   size=num_particles_ok)  # random set of particles

            for ind_t in range(number_of_time_intervals):

                v = data_dict['v'][inds_particles, ind_t]
                vt = data_dict['v_transverse'][inds_particles, ind_t]
                vz = data_dict['v_axial'][inds_particles, ind_t]
                vz0 = data_dict['v_axial'][inds_particles, 0]
                B = data_dict['B'][inds_particles, ind_t]
                B0 = data_dict['B'][inds_particles, 0]
                B_max = B0 * Rm

                # initialize
                if ind_t == 0:
                    inds_bins_ini = [np.nan for _ in range(num_particles_ok)]

                particles_counter_mat = np.zeros([N_theta, N_theta])

                for ind_p in inds_particles:
                    if (vt[ind_p] / v[ind_p]) ** 2 < B[ind_p] / B_max[ind_p]:
                        if vz[ind_p] > 0:
                            # in right loss cone
                            ind_bin_fin = 0
                        else:
                            # in left loss cone
                            ind_bin_fin = 2
                    else:
                        # outside of loss cone
                        ind_bin_fin = 1

                    if ind_t == 0:
                        inds_bins_ini[ind_p] = ind_bin_fin
                    ind_bin_ini = inds_bins_ini[ind_p]

                    particles_counter_mat[ind_bin_ini, ind_bin_fin] += 1
                # print(f'particles_counter_mat={particles_counter_mat}')

                if ind_t == 0:
                    N0 = copy.deepcopy(np.diag(particles_counter_mat))

                particles_counter_mat_4d[:, :, ind_t, ind_boot] = particles_counter_mat

            # divide all densities by the parent initial density
            for ind_t in range(number_of_time_intervals):
                for ind_bin in range(N_theta):
                    particles_counter_mat_4d[ind_bin, :, ind_t, ind_boot] /= (1.0 * N0[ind_bin])

        # compile the results from the bootstrap procedure
        for process_name, process_index_pair in zip(process_names, process_index_pairs):
            pi, pj = process_index_pair[0], process_index_pair[1]
            compiled_dict['N_' + process_name + '_curve_mean'] = np.mean(particles_counter_mat_4d[pi, pj, :, :], axis=1)
            compiled_dict['N_' + process_name + '_curve_std'] = np.std(particles_counter_mat_4d[pi, pj, :, :], axis=1)

            num_t_inds_avg = 5
            inds_t_avg = range(number_of_time_intervals - num_t_inds_avg,
                               number_of_time_intervals)  # only the last few indices
            compiled_dict['N_' + process_name + '_end'] = np.mean(particles_counter_mat_4d[pi, pj, inds_t_avg, 0])
            compiled_dict['N_' + process_name + '_end_std'] = np.std(particles_counter_mat_4d[pi, pj, inds_t_avg, :])

        #### plot bootstrap curves

        fig, ax = plt.subplots(1, 1,
                               figsize=(9, 6),
                               )


        def plot_line(x, y, label, color, linestyle='-', linewidth=2):
            ax.plot(x, y, color=color, linestyle=linestyle, label=label, linewidth=linewidth)


        for ind_boot in range(num_bootstrap_samples):

            if ind_boot == 0:
                linestyle1 = '-'
                linewidth1 = 2
            else:
                linestyle1 = '--'
                linewidth1 = 1
            linestyle2 = '--'
            # plot_saturation = True
            plot_saturation = False

            N_rc_tilde = particles_counter_mat_4d[0, 1, :, ind_boot]
            label = '$\\bar{N}_{rc}$' if ind_boot == 0 else None
            ax.plot(t_array, N_rc_tilde, label=label, color='b', linestyle=linestyle1, linewidth=linewidth1)

            N_cr_tilde = particles_counter_mat_4d[1, 0, :, ind_boot]
            label = '$\\bar{N}_{cr}$' if ind_boot == 0 else None
            ax.plot(t_array, N_cr_tilde, label=label, color='g', linestyle=linestyle1, linewidth=linewidth1)

            N_lc_tilde = particles_counter_mat_4d[2, 1, :, ind_boot]
            label = '$\\bar{N}_{lc}$' if ind_boot == 0 else None
            ax.plot(t_array, N_lc_tilde, label=label, color='r', linestyle=linestyle1, linewidth=linewidth1)

            N_cl_tilde = particles_counter_mat_4d[1, 2, :, ind_boot]
            label = '$\\bar{N}_{cl}$' if ind_boot == 0 else None
            ax.plot(t_array, N_cl_tilde, label=label, color='orange', linestyle=linestyle1, linewidth=linewidth1)

            N_rl_tilde = particles_counter_mat_4d[0, 2, :, ind_boot]
            label = '$\\bar{N}_{rl}$' if ind_boot == 0 else None
            ax.plot(t_array, N_rl_tilde, label=label, color='k', linestyle=linestyle1, linewidth=linewidth1)

            N_lr_tilde = particles_counter_mat_4d[2, 0, :, ind_boot]
            label = '$\\bar{N}_{lr}$' if ind_boot == 0 else None
            ax.plot(t_array, N_lr_tilde, label=label, color='brown', linestyle=linestyle1, linewidth=linewidth1)

        ax.set_xlabel('t/($l/v_{th}$)', fontsize=12)

        # ax.legend(loc='upper left', fontsize=20)
        # ax.legend(loc='upper left', fontsize=15)
        ax.legend(loc='lower right', fontsize=15)
        ax.grid(True)

        title = set_name
        ax.set_title(title)

        # fig.set_tight_layout({'pad': 0.5, 'rect': (0, 0, 1, 0.95)})
        fig.set_layout_engine(layout='tight')

        #### Plot with fill_between
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), )
        linestyle1 = '-'
        linewidth1 = 2


        def plot_fill_between(x, y_mean, y_std, label, color, linestyle='-', linewidth=2, num_sigmas=2):
            ax.plot(x, y_mean, color=color, linestyle=linestyle, label=label, linewidth=linewidth)
            ax.fill_between(x, y_mean + num_sigmas * y_std, y_mean - num_sigmas * y_std, color=color, alpha=0.5)


        N_rc_tilde = particles_counter_mat_4d[0, 1, :, :]
        label = '$\\bar{N}_{rc}$'
        plot_fill_between(t_array, np.mean(N_rc_tilde, axis=1), np.std(N_rc_tilde, axis=1), label=label, color='b',
                          linestyle=linestyle1, linewidth=linewidth1)

        N_cr_tilde = particles_counter_mat_4d[1, 0, :, :]
        label = '$\\bar{N}_{cr}$'
        plot_fill_between(t_array, np.mean(N_cr_tilde, axis=1), np.std(N_cr_tilde, axis=1), label=label, color='g',
                          linestyle=linestyle1, linewidth=linewidth1)

        N_lc_tilde = particles_counter_mat_4d[2, 1, :, :]
        label = '$\\bar{N}_{lc}$'
        plot_fill_between(t_array, np.mean(N_lc_tilde, axis=1), np.std(N_lc_tilde, axis=1), label=label, color='r',
                          linestyle=linestyle1, linewidth=linewidth1)

        N_cl_tilde = particles_counter_mat_4d[1, 2, :, :]
        label = '$\\bar{N}_{cl}$'
        plot_fill_between(t_array, np.mean(N_cl_tilde, axis=1), np.std(N_cl_tilde, axis=1), label=label, color='orange',
                          linestyle=linestyle1, linewidth=linewidth1)

        N_rl_tilde = particles_counter_mat_4d[0, 2, :, :]
        label = '$\\bar{N}_{rl}$'
        plot_fill_between(t_array, np.mean(N_rl_tilde, axis=1), np.std(N_rl_tilde, axis=1), label=label, color='k',
                          linestyle=linestyle1, linewidth=linewidth1)

        N_lr_tilde = particles_counter_mat_4d[2, 0, :, :]
        label = '$\\bar{N}_{lr}$'
        plot_fill_between(t_array, np.mean(N_lr_tilde, axis=1), np.std(N_lr_tilde, axis=1), label=label, color='brown',
                          linestyle=linestyle1, linewidth=linewidth1)

        ax.set_xlabel('t/($l/v_{th}$)', fontsize=12)
        ax.legend(loc='lower right', fontsize=15)
        ax.grid(True)
        ax.set_title(title)
        fig.set_layout_engine(layout='tight')

# plt.figure()
# for i in range(100):
#     plt.plot(data_dict['r'][i])
#     # plt.plot(data_dict['r'][i] - data_dict['r'][i][0])
