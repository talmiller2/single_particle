import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_thermal_velocity, get_cyclotron_angular_frequency

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set48_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
save_dir += '/set49_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'

# RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
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
# gas_name_list += ['deuterium']
# gas_name_list += ['DT_mix']
gas_name_list += ['tritium']

select_alpha_list = []
select_beta_list = []
set_name_list = []

# select_alpha_list += [1]
# select_beta_list += [0]
# set_name_list += ['noRF']
#
# select_alpha_list += [0.64]
# select_beta_list += [-1.8]
# set_name_list += ['1']

# select_alpha_list += [0.7]
# select_beta_list += [-0.8]
# set_name_list += ['2']

# select_alpha_list += [1.06]
# select_beta_list += [-1.8]
# set_name_list += ['3']

# select_alpha_list += [1.12]
# select_beta_list += [1.4]
# set_name_list += ['4']

# select_alpha_list += [0.88]
# select_beta_list += [0.0]
# set_name_list += ['5']

# select_alpha_list += [1.0]
# select_beta_list += [-1.8]
# set_name_list += ['6']

select_alpha_list += [0.88]
select_beta_list += [-1.8]
set_name_list += ['7']

plot_theta_trajectories = False
# plot_theta_trajectories = True

# plot_axial_trajectories = False
plot_axial_trajectories = True

# plot_radial_trajectories = False
plot_radial_trajectories = True

plot_population_tracker = False
# plot_population_tracker = True


use_RF = True
# use_RF = False
with_RF_xy_corrections = True
# with_RF_xy_corrections = False
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
sigma_r0 = 0.1

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
            if with_RF_xy_corrections == False:
                set_name += '_woxyRFcor'
        set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
        if sigma_r0 > 0:
            set_name += '_sigmar' + str(sigma_r0)
        set_name += '_' + gas_name
        print(set_name)

        save_dir_curr = save_dir + set_name

        # load runs data
        data_dict_file = save_dir_curr + '.pickle'
        with open(data_dict_file, 'rb') as fid:
            data_dict = pickle.load(fid)
        settings_file = save_dir + 'settings.pickle'
        with open(settings_file, 'rb') as fid:
            settings = pickle.load(fid)
        field_dict_file = save_dir + 'field_dict.pickle'
        with open(field_dict_file, 'rb') as fid:
            field_dict = pickle.load(fid)

        num_particles = len(data_dict['t'])
        # num_particles = 1
        # num_particles = 3
        # num_particles = 50
        # num_particles = 200
        # num_particles = 1000

        # define v_th ref
        _, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
        v_th_ref = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])

        # divide the phase space by the angle
        theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))

        ### plot the theretical resonance points
        # _, _, mi, _, Z_ion = define_plasma_parameters(gas_name=gas_name)
        _, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
        q = Z_ion * settings['e']  # Coulomb
        omega_cyc_0 = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)
        omega_RF = alpha * field_dict['omega_cyclotron']
        omega_RF_over_omega_cyc_0 = omega_RF / omega_cyc_0
        k_RF = 2 * np.pi / field_dict['l'] * beta
        if k_RF != 0:
            v_RF = omega_RF / (2 * np.pi * k_RF)

        if use_RF:
            RF_str = ('$\\omega_{RF}/\\omega_{cyc,T}$=' + '{:.2f}'.format(omega_RF_over_omega_cyc_0)
                      + ', $k_{RF}/2\\pi$=' + '{:.2f}'.format(k_RF / (2 * np.pi)))
        else:
            RF_str = 'no RF'

        ## plot
        if plot_theta_trajectories:
            fig, axs = plt.subplots(num=fig_num, nrows=1, ncols=2, figsize=(12, 6))
        if plot_axial_trajectories:
            fig_num += 1
            fig2, axs2 = plt.subplots(num=fig_num, nrows=1, ncols=2, figsize=(12, 6))
        if plot_radial_trajectories:
            fig_num += 1
            fig3, axs3 = plt.subplots(num=fig_num, nrows=1, ncols=2, figsize=(12, 6))

        num_particles_LC = 0

        for ind_p in range(num_particles):

            t = np.array(data_dict['t'][ind_p])

            # if len(t) < 15:
            if ind_p == 44:
                print('index of particle that ended prematurely:', ind_p)

                t /= (field_dict['l'] / v_th_ref)

                r = np.array(data_dict['r'][ind_p])
                z = np.array(data_dict['z'][ind_p])
                v = np.array(data_dict['v'][ind_p])

                v0 = data_dict['v'][ind_p][0]
                vt = np.array(data_dict['v_transverse'][ind_p])
                vt0 = data_dict['v_transverse'][ind_p][0]
                vz = np.array(data_dict['v_axial'][ind_p])
                vz0 = data_dict['v_axial'][ind_p][0]
                theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
                Bz = np.array(data_dict['Bz'][ind_p])
                Bz0 = data_dict['Bz'][ind_p][0]
                vt_adjusted = vt * np.sqrt(
                    Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

                det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
                inds_positive = np.where(det > 0)[0]
                vz_adjusted = np.zeros(len(vz))
                vz_adjusted[inds_positive] = np.sign(vz0) * np.sqrt(det[inds_positive])
                # vz_adjusted[inds_positive] = np.sign(vz) * np.sqrt(det[inds_positive])
                theta_adjusted = np.mod(360 / (2 * np.pi) * np.arctan(vt_adjusted / vz_adjusted), 180)

                if plot_population_tracker:
                    # initialize tracking 3 populations as a function of time according to axial location
                    if ind_p == 0:
                        populations_counter_mat = np.zeros([len(t), 3, 3])
                        z_mirror_max = settings['l']
                        z_mirror_min = 0
                        labels_dict = {0: 'right', 2: 'left', 1: 'trap'}

                    if theta[0] < theta_LC:
                        ind_ini = 0  # right loss cone
                        num_particles_LC += 1
                    elif theta[0] > 180 - theta_LC:
                        ind_ini = 2  # left loss cone
                    else:
                        ind_ini = 1  # trapped

                    for ind_t in range(len(t)):
                        if z[ind_t] > z_mirror_max:
                            ind_fin = 0
                        elif z[ind_t] < z_mirror_min:
                            ind_fin = 2
                        else:
                            ind_fin = 1
                        populations_counter_mat[ind_t, ind_ini, ind_fin] += 1

                if plot_theta_trajectories or plot_axial_trajectories:
                    ## color the lines according to a metric
                    # metric = abs((max(vz_adjusted) - min(vz_adjusted)) / vz_adjusted[0])
                    # metric = (max(vz_adjusted) - min(vz_adjusted)) / v_th_ref
                    # metric = (max(theta_adjusted) - min(theta_adjusted)) / 180 * 2
                    # metric = (max(vz_adjusted) - min(vz_adjusted)) / (2 * v_th_ref)
                    metric = theta[0] / 180
                    color = cm.rainbow(metric)

                if plot_theta_trajectories:
                    for ind_ax, ax in enumerate(axs):

                        # if ind_p == num_particles - 1 and k_RF != 0
                        #     ax.hlines(v_RF / v_th_ref, 0, max(t), colors='k', linestyles='dashed', linewidth=2)
                        ax.hlines(theta_LC, 0, max(t), colors='k', linestyles='dashed', linewidth=2)
                        ax.hlines(180 - theta_LC, 0, max(t), colors='k', linestyles='dashed', linewidth=2)

                        alpha_lines = 0.4
                        plot_particle = False
                        if ind_ax == 0 and (theta[0] < theta_LC or theta[0] > 180 - theta_LC):
                            plot_particle = True
                        if ind_ax == 1 and (theta[0] > theta_LC and theta[0] < 180 - theta_LC):
                            plot_particle = True

                        if plot_particle:
                            ax.plot(t,
                                    # vz_adjusted / v_th_ref,
                                    theta_adjusted,
                                    color=color,
                                    alpha=alpha_lines,
                                    )

                if plot_axial_trajectories:
                    for ind_ax, ax in enumerate(axs2):
                        ax.hlines(0, 0, max(t), colors='k', linestyles='dashed', linewidth=2)
                        ax.hlines(field_dict['l'], 0, max(t), colors='k', linestyles='dashed', linewidth=2)

                        alpha_lines = 0.4

                        plot_particle = False
                        if ind_ax == 0 and (theta[0] < theta_LC or theta[0] > 180 - theta_LC):
                            # if ind_ax == 0 and theta[0] < theta_LC:
                            # if ind_ax == 0 and theta[0] > theta_LC:
                            plot_particle = True
                        if ind_ax == 1 and (theta[0] > theta_LC and theta[0] < 180 - theta_LC):
                            # if ind_ax == 1 and theta[0] > 180 - theta_LC:
                            # if ind_ax == 1 and theta[0] < 180 - theta_LC:
                            plot_particle = True

                        if plot_particle:
                            ax.plot(t,
                                    z,
                                    color=color,
                                    alpha=alpha_lines,
                                    )

                if plot_radial_trajectories:
                    for ind_ax, ax in enumerate(axs3):
                        alpha_lines = 0.4

                        plot_particle = False
                        if ind_ax == 0 and (theta[0] < theta_LC or theta[0] > 180 - theta_LC):
                            plot_particle = True
                        if ind_ax == 1 and (theta[0] > theta_LC and theta[0] < 180 - theta_LC):
                            plot_particle = True

                        if plot_particle:
                            ax.plot(t,
                                    r,
                                    color=color,
                                    alpha=alpha_lines,
                                    )

        if plot_theta_trajectories:
            for ind_ax, ax in enumerate(axs):
                if ind_ax == 0:
                    title_prefix = 'initially in loss cones '
                elif ind_ax == 1:
                    title_prefix = 'initially trapped '

                ax.set_title(title_prefix + RF_str, fontsize=12)
                # ax.set_xlabel('t [s]', fontsize=12)
                ax.set_xlabel('t/($l/v_{th,T}$)', fontsize=12)
                # ax.set_ylabel('$\\tilde{v}_z / v_{th,T}$', fontsize=12)
                # ax.set_ylim([-2.5, 2.5])
                ax.set_ylabel('$\\tilde{\\theta} [deg]$', fontsize=12)
                ax.set_ylim([0, 180])
                ax.grid(True)
            fig.set_layout_engine(layout='tight')

        if plot_axial_trajectories:
            for ind_ax, ax in enumerate(axs2):
                if ind_ax == 0:
                    title_prefix = 'initially in loss cones '
                elif ind_ax == 1:
                    title_prefix = 'initially trapped '

                ax.set_title(title_prefix + RF_str, fontsize=12)
                # ax.set_xlabel('t [s]', fontsize=12)
                ax.set_xlabel('t/($l/v_{th,T}$)', fontsize=12)
                ax.set_ylabel('z [m]', fontsize=12)
                # ax.set_ylim([-field_dict['l'], 2 * field_dict['l']])
                ax.set_ylim([-3 * field_dict['l'], 4 * field_dict['l']])
                ax.grid(True)
            fig2.set_layout_engine(layout='tight')

        if plot_radial_trajectories:
            for ind_ax, ax in enumerate(axs3):
                if ind_ax == 0:
                    title_prefix = 'initially in loss cones '
                elif ind_ax == 1:
                    title_prefix = 'initially trapped '

                ax.set_title(title_prefix + RF_str, fontsize=12)
                # ax.set_xlabel('t [s]', fontsize=12)
                ax.set_xlabel('t/($l/v_{th,T}$)', fontsize=12)
                ax.set_ylabel('r [m]', fontsize=12)
                # ax.set_ylim([-field_dict['l'], 2 * field_dict['l']])
                # ax.set_ylim([-3 * field_dict['l'], 4 * field_dict['l']])
                ax.grid(True)
            fig2.set_layout_engine(layout='tight')

        if plot_population_tracker:
            # plot populations tracker
            fig_num += 1
            fig3, ax3 = plt.subplots(num=fig_num, nrows=1, ncols=1)
            fig = fig3
            ax = ax3
            populations_counter_mat /= num_particles_LC  # normalize
            for ind_ini in range(3):
                for ind_fin in range(3):
                    if (ind_ini, ind_fin) == (0, 1):
                        color = 'b'
                    elif (ind_ini, ind_fin) == (1, 0):
                        color = 'g'
                    elif (ind_ini, ind_fin) == (2, 1):
                        color = 'r'
                    elif (ind_ini, ind_fin) == (1, 2):
                        color = 'orange'
                    elif (ind_ini, ind_fin) == (0, 2):
                        color = 'cyan'
                    elif (ind_ini, ind_fin) == (2, 0):
                        color = 'yellow'
                    else:
                        color = 'k'

                    if ind_fin != ind_ini:
                        ax.plot(t, populations_counter_mat[:, ind_ini, ind_fin],
                                label=labels_dict[ind_ini] + '->' + labels_dict[ind_fin], color=color, alpha=0.7)

            # difference plots
            ind_ini = 0
            ind_fin = 1
            ax.plot(t, populations_counter_mat[:, ind_ini, ind_fin] - populations_counter_mat[:, ind_fin, ind_ini],
                    '--b',
                    linewidth=3,
                    label='(' + labels_dict[ind_ini] + '->' + labels_dict[ind_fin] + ') - (' + labels_dict[
                        ind_fin] + '->' +
                          labels_dict[ind_ini] + ')')
            ind_ini = 2
            ind_fin = 1
            ax.plot(t, populations_counter_mat[:, ind_ini, ind_fin] - populations_counter_mat[:, ind_fin, ind_ini],
                    '--r',
                    linewidth=3,
                    label='(' + labels_dict[ind_ini] + '->' + labels_dict[ind_fin] + ') - (' + labels_dict[
                        ind_fin] + '->' +
                          labels_dict[ind_ini] + ')')

            ax.legend()
            ax.set_xlabel('t/($l/v_{th,T}$)', fontsize=12)
            # ax.set_ylabel('# particles', fontsize=12)
            ax.set_ylabel('$N/N_{cone}$', fontsize=12)
            ax.set_title(RF_str, fontsize=12)
            ax.grid(True)
            fig.set_layout_engine(layout='tight')
