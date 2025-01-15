import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_thermal_velocity, get_cyclotron_angular_frequency

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

plt.close('all')

# plot_saturation_lines = True
plot_saturation_lines = False

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set48_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set49_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
save_dir += '/set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 25  # kV/m
# E_RF_kVm = 25  # kV/m
E_RF_kVm = 50  # kV/m
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

# select_alpha_list += [0.64]
# select_beta_list += [-1.8]
# set_name_list += ['1']

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
select_beta_list += [0.0]
set_name_list += ['5']

# select_alpha_list += [1.48]
# select_beta_list += [-1.4]
# set_name_list += ['TEST']

use_RF = True
# use_RF = False
# with_kr_correction = False
with_kr_correction = True
# induced_fields_factor = 1
# induced_fields_factor = 0.5
# induced_fields_factor = 0.1
# induced_fields_factor = 0.01
induced_fields_factor = 0
# time_step_tau_cyclotron_divisions = 20
# time_step_tau_cyclotron_divisions = 40
time_step_tau_cyclotron_divisions = 50
# time_step_tau_cyclotron_divisions = 80
# sigma_r0 = 0
sigma_r0 = 0.05
# sigma_r0 = 0.1
radial_distribution = 'uniform'

fig_num = 0

for gas_name in gas_name_list:
    for ind_set in range(len(set_name_list)):
        fig_num += 1

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
        len_t_expected = 30
        num_particles = len(data_dict['t'])
        inds_ok = []
        for ind_particle, t in enumerate(data_dict['t']):
            if len(t) == len_t_expected:
                inds_ok += [ind_particle]
            # inds_ok += [ind_particle]
        percent_ok = len(inds_ok) / num_particles * 100
        print('percent_ok:', percent_ok)
        for key in data_dict.keys():
            # data_dict[key] = np.array([data_dict[key][i][0:30] for i in inds_ok])
            data_dict[key] = np.array([data_dict[key][i] for i in inds_ok])

        # divide the phase space by the angle
        theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
        N_theta_LC = 1
        N_theta_T = 1
        N_theta = 2 * N_theta_LC + N_theta_T
        dtheta_LC = theta_LC / N_theta_LC
        dtheta_T = (180 - 2 * theta_LC) / N_theta_T
        theta_bins_max_list = [dtheta_LC]
        for i in range(N_theta_LC - 1):
            theta_bins_max_list += [theta_bins_max_list[-1] + dtheta_LC]
        for i in range(N_theta_T):
            theta_bins_max_list += [theta_bins_max_list[-1] + dtheta_T]
        for i in range(N_theta_LC):
            theta_bins_max_list += [theta_bins_max_list[-1] + dtheta_LC]
        theta_bins_min_list = [0] + theta_bins_max_list[:-1]

        # number_of_time_intervals = data_dict['t'].shape[1]
        number_of_time_intervals = len(data_dict['t'][0])

        particles_counter_mat_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])

        from matplotlib import cm

        colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
        # colors = ['b', 'g', 'r']
        # colors = ['r', 'g', 'b']

        for ind_t in range(number_of_time_intervals):
            # for ind_t in [0, 10]:
            # for ind_t in [0, 1]:
            # for ind_t in [0, 10, 20]:
            #     print(ind_t)

            # inds_particles = range(data_dict['t'].shape[0])
            inds_particles = range(len(data_dict['t']))
            # inds_particles = range(1000)
            # inds_particles = [0, 1, 2]
            # inds_particles = range(1001)
            if ind_t == 0:
                print('num particles = ' + str(len(inds_particles)))

            v = data_dict['v'][inds_particles, ind_t]
            v0 = data_dict['v'][inds_particles, 0]
            vt = data_dict['v_transverse'][inds_particles, ind_t]
            vt0 = data_dict['v_transverse'][inds_particles, 0]
            vz = data_dict['v_axial'][inds_particles, ind_t]
            vz0 = data_dict['v_axial'][inds_particles, 0]
            theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
            Bz = data_dict['Bz'][inds_particles, ind_t]
            Bz0 = data_dict['Bz'][inds_particles, 0]
            vt_adjusted = vt * np.sqrt(
                Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

            det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
            inds_positive = np.where(det > 0)[0]

            vz_adjusted = np.zeros(len(inds_particles))
            # vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])
            vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(
                det[inds_positive])  # updated criterion

            theta_adjusted = 90.0 * np.ones(len(inds_particles))  # if det<0 particle probably close to vz=0
            theta_adjusted[inds_positive] = np.mod(
                360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)

            color = colors[ind_t]

            if ind_t == 0:
                inds_bins_ini = []
            particles_counter_mat = np.zeros([N_theta, N_theta])

            for ind_p in inds_particles:
                # if not cancelled_particles[ind_p]:
                theta_curr = theta_adjusted[ind_p]
                ind_bin_fin = [k for k, (t1, t2) in enumerate(zip(theta_bins_min_list, theta_bins_max_list))
                               if theta_curr > t1 and theta_curr <= t2][0]
                if ind_t == 0:
                    inds_bins_ini += [ind_bin_fin]

                ind_bin_ini = inds_bins_ini[ind_p]

                particles_counter_mat[ind_bin_ini, ind_bin_fin] += 1

            if ind_t == 0:
                N0 = copy.deepcopy(np.diag(particles_counter_mat))

            particles_counter_mat_3d[:, :, ind_t] = particles_counter_mat

        # divide all densities by the parent initial density
        for ind_t in range(number_of_time_intervals):
            for ind_bin in range(N_theta):
                particles_counter_mat_3d[ind_bin, :, ind_t] /= (1.0 * N0[ind_bin])

        t_array = data_dict['t'][0]
        # t_array /= settings['l'] / settings['v_th']
        # t_array /= settings['l'] / settings['v_th_for_cyc']

        _, _, mi, _, Z_ion = define_plasma_parameters(gas_name=gas_name)
        v_th_curr = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])
        t_array /= settings['l'] / v_th_curr

        # define v_th ref
        _, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
        v_th_ref = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])
        q = Z_ion * settings['e']  # Coulomb
        omega_cyc_0 = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)
        omega_RF = alpha * field_dict['omega_cyclotron']
        omega_RF_over_omega_cyc_0 = omega_RF / omega_cyc_0
        k_RF = 2 * np.pi / field_dict['l'] * beta
        if k_RF != 0:
            v_RF = omega_RF / (2 * np.pi * k_RF)

        # t_array /= settings['l'] / v_th_ref

        inds_t_array = range(len(t_array))
        fig, ax = plt.subplots(1, 1,
                               figsize=(9, 6),
                               )
        # fig.suptitle(title)

        ## calculate the saturation value to estimate the rate
        # inds_t_saturation = range(15, 31) # for 2023 paper
        # inds_t_saturation = range(15, 29)
        inds_t_saturation = range(15, 30)  # for 2024 paper


        def plot_line(x, y, label, color, linestyle='-', linewidth=2, plot_saturation=False, saturation_linestyl=':'):
            if plot_saturation:
                saturation_value = np.mean(y[inds_t_saturation])
                label += '=' + '{:.3f}'.format(saturation_value)
                ax.hlines(saturation_value, x[inds_t_saturation[0]], x[inds_t_saturation[-1]],
                          color=color, linewidth=linewidth, linestyle=saturation_linestyl)

            ax.plot(x, y, color=color, linestyle=linestyle, label=label, linewidth=linewidth)


        linestyle1 = '-'
        linestyle2 = '--'
        plot_saturation = True

        N_rc_tilde = particles_counter_mat_3d[0, 1, :]
        plot_line(t_array, N_rc_tilde, label='$\\bar{N}_{rc}$', color='b', linestyle=linestyle1,
                  plot_saturation=plot_saturation)

        N_cr_tilde = particles_counter_mat_3d[1, 0, :]
        plot_line(t_array, N_cr_tilde, label='$\\bar{N}_{cr}$', color='g', linestyle=linestyle1,
                  plot_saturation=plot_saturation)

        N_lc_tilde = particles_counter_mat_3d[2, 1, :]
        plot_line(t_array, N_lc_tilde, label='$\\bar{N}_{lc}$', color='r', linestyle=linestyle1,
                  plot_saturation=plot_saturation)

        N_cl_tilde = particles_counter_mat_3d[1, 2, :]
        plot_line(t_array, N_cl_tilde, label='$\\bar{N}_{cl}$', color='orange', linestyle=linestyle1,
                  plot_saturation=plot_saturation)

        N_rl_tilde = particles_counter_mat_3d[0, 2, :]
        plot_line(t_array, N_rl_tilde, label='$\\bar{N}_{rl}$', color='k', linestyle=linestyle1,
                  plot_saturation=plot_saturation)

        N_lr_tilde = particles_counter_mat_3d[2, 0, :]
        plot_line(t_array, N_lr_tilde, label='$\\bar{N}_{lr}$', color='brown', linestyle=linestyle1,
                  plot_saturation=plot_saturation)

        # LC_ini_fraction = np.sin(np.arcsin(field_dict['Rm'] ** (-0.5)) / 2) ** 2
        # trapped_ini_fraction = 1 - 2 * LC_ini_fraction
        #
        # cone_escape_rate_R = (N_rc_tilde * LC_ini_fraction - N_cr_tilde * trapped_ini_fraction) / LC_ini_fraction
        # plot_line(t_array, cone_escape_rate_R, label='$(N_{rc}-N_{cr})/N_{cone}$', color='b', linestyle=linestyle2,
        #           plot_saturation=plot_saturation)
        #
        # cone_escape_rate_L = (N_lc_tilde * LC_ini_fraction - N_cl_tilde * trapped_ini_fraction) / LC_ini_fraction
        # plot_line(t_array, cone_escape_rate_L, label='$(N_{lc}-N_{cl})/N_{cone}$', color='r', linestyle=linestyle2,
        #           plot_saturation=plot_saturation)

        # ax.set_xlabel('$t \\cdot v_{th} / l$')
        # ax.set_xlabel('t/($l/v_{th,T}$)', fontsize=12)
        ax.set_xlabel('t/($l/v_{th}$)', fontsize=12)
        # ax.set_xlabel('$t / \\tau_{th}$',
        #               fontsize=20)
        # ax.set_ylim([0, 0.9])
        # ax.set_ylim([0, 1.0])
        # ax.set_xlim([0, 1.75])

        # ax.legend(loc='upper left', fontsize=20)
        # ax.legend(loc='upper left', fontsize=15)
        ax.legend(loc='lower right', fontsize=15)
        ax.grid(True)

        # text = '(a)'
        if gas_name == 'deuterium':
            gas_name_shorthand = 'D'
        if gas_name == 'tritium':
            gas_name_shorthand = 'T'
        # text = '(' + gas_name_shorthand + ',' + RF_set_name + ')'
        # text = RF_set_name + ' (' + gas_name_shorthand + ')'
        E_ratio_mean = np.mean(data_dict['v'][inds_particles, -1] ** 2) / np.mean(
            data_dict['v'][inds_particles, 0] ** 2)
        # E_ratio_std = np.std(data_dict['v'][inds_particles, -1] ** 2) / np.std(data_dict['v'][inds_particles, 0] ** 2)
        text = '$\\bar{E}_{fin}/\\bar{E}_{ini}=$' + '{:.2f}'.format(E_ratio_mean)
        # text += ', $\\sigma\\left(E_{fin}\\right)/\\sigma\\left(E_{ini}\\right)=$' + '{:.2f}'.format(E_ratio_std)
        plt.text(
            # 0.99, 0.98, text,
            0.02, 0.98,
            text,
            # fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
                 fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
            # horizontalalignment='right', verticalalignment='top',
            horizontalalignment='left', verticalalignment='top',
            color='k',
                 transform=fig.axes[0].transAxes)

        # title = gas_name + ': '
        # if use_RF:
        #     title += ('$\\omega_{RF}/\\omega_{cyc,T}$=' + '{:.2f}'.format(omega_RF_over_omega_cyc_0)
        #              + ', $k_{RF}/2\\pi$=' + '{:.2f}'.format(k_RF / (2 * np.pi)))
        # else:
        #     title += 'no RF'
        title = set_name
        # ax.set_title(gas_name)
        ax.set_title(title)

        # fig.set_tight_layout({'pad': 0.5, 'rect': (0, 0, 1, 0.95)})
        fig.set_layout_engine(layout='tight')

        ## save plots to file
        # save_fig_dir = '../../../Papers/texts/paper2022/pics/'
        # file_name = 'saturation_set_' + gas_name_shorthand + '_' + RF_set_name
        # if RF_type == 'magnetic_transverse':
        #     file_name = 'BRF_' + file_name
        # # file_name += '_mod'
        # beingsaved = plt.gcf()
        # beingsaved.savefig(save_fig_dir + file_name + '.eps', format='eps')
