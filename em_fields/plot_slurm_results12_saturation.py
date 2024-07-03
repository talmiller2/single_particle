import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np

from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_thermal_velocity

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

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
# save_dir += '/set36_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set37_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set38_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
save_dir += '/set39_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 25  # kV/m
E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

# RF_type = 'magnetic_transverse'
# B_RF = 0.01  # T
# B_RF = 0.02  # T
# B_RF = 0.04  # T
# B_RF = 0.05  # T
# B_RF = 0.1  # T

gas_name_list = []
gas_name_list += ['deuterium']
# gas_name_list += ['DT_mix']
gas_name_list += ['tritium']

use_RF = True
# use_RF = False

alpha_loop_list = np.round(np.linspace(0.5, 1.5, 21), 2)  # set37, 39
beta_loop_list = np.round(np.linspace(-10, 10, 21), 2)

select_alpha_list = []
select_beta_list = []
set_name_list = []

select_alpha_list += [1.4]
select_beta_list += [3.0]
set_name_list += ['1']

select_alpha_list += [1.0]
select_beta_list += [-3.0]
set_name_list += ['2']

select_alpha_list += [0.7]
select_beta_list += [-3.0]
set_name_list += ['3']

select_alpha_list += [0.55]
select_beta_list += [-7.0]
set_name_list += ['4']

ind_set = 0
# ind_set = 1
# ind_set = 2
# ind_set = 3

alpha = select_alpha_list[ind_set]
beta = select_beta_list[ind_set]
RF_set_name = set_name_list[ind_set]
# RF_set_name = '$\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)

for gas_name in gas_name_list:

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

    # for i in range(100):
    #     print(len(data_dict['t'][i]), data_dict['t'][i][-1], data_dict['r'][i][-1])

    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key])

    # # filter out the particles that ended prematurely
    # num_particles = len(data_dict['t'])
    # inds_ok = []
    # for ind_particle, t in enumerate(data_dict['t']):
    #     if len(t) == 30:
    #         inds_ok += [ind_particle]
    # percent_ok = len(inds_ok) / num_particles * 100
    # for key in data_dict.keys():
    #     data_dict[key] = np.array([data_dict[key][i][0:30] for i in inds_ok])

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

    number_of_time_intervals = data_dict['t'].shape[1]

    particles_counter_mat_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])
    particles_counter_mat2_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])

    from matplotlib import cm

    colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
    # colors = ['b', 'g', 'r']
    # colors = ['r', 'g', 'b']

    for ind_t in range(number_of_time_intervals):
        # for ind_t in [0, 10]:
        # for ind_t in [0, 1]:
        # for ind_t in [0, 10, 20]:
        #     print(ind_t)

        inds_particles = range(data_dict['t'].shape[0])
        # inds_particles = range(len(data_dict['t']))
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
        vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])

        theta_adjusted = 90.0 * np.ones(len(inds_particles))
        theta_adjusted[inds_positive] = np.mod(
            360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)

        color = colors[ind_t]

        # track if a particle left the population, and then cancel counting it for the following times
        if ind_t == 0:
            particles_counter_mat = np.zeros([N_theta, N_theta])
            inds_bins_ini = []
            cancelled_particles = np.zeros(len(inds_particles))
        particles_counter_mat2 = np.zeros([N_theta, N_theta])

        for ind_p in inds_particles:
            # if not cancelled_particles[ind_p]:
            theta_curr = theta_adjusted[ind_p]
            ind_bin_fin = [k for k, (t1, t2) in enumerate(zip(theta_bins_min_list, theta_bins_max_list))
                           if theta_curr > t1 and theta_curr <= t2][0]
            if ind_t == 0:
                inds_bins_ini += [ind_bin_fin]
                particles_counter_mat[ind_bin_fin, ind_bin_fin] += 1

            ind_bin_ini = inds_bins_ini[ind_p]

            particles_counter_mat2[ind_bin_ini, ind_bin_fin] += 1
            if ind_bin_fin != ind_bin_ini:
                if not cancelled_particles[ind_p]:
                    particles_counter_mat[ind_bin_ini, ind_bin_ini] -= 1
                    particles_counter_mat[ind_bin_ini, ind_bin_fin] += 1
                    cancelled_particles[ind_p] = 1

        if ind_t == 0:
            N0 = copy.deepcopy(np.diag(particles_counter_mat))

        particles_counter_mat_3d[:, :, ind_t] = particles_counter_mat
        particles_counter_mat2_3d[:, :, ind_t] = particles_counter_mat2

    # divide all densities by the parent initial density
    for ind_t in range(number_of_time_intervals):
        for ind_bin in range(N_theta):
            particles_counter_mat_3d[ind_bin, :, ind_t] /= (1.0 * N0[ind_bin])
            particles_counter_mat2_3d[ind_bin, :, ind_t] /= (1.0 * N0[ind_bin])
    particles_counter_mat2_for_fit_3d = copy.deepcopy(particles_counter_mat2_3d)

    t_array = data_dict['t'][0]
    # t_array /= settings['l'] / settings['v_th']
    # t_array /= settings['l'] / settings['v_th_for_cyc']
    # define v_th ref
    _, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
    v_th_ref = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])
    t_array /= settings['l'] / v_th_ref

    colors = cm.rainbow(np.linspace(0, 1, N_theta))
    nu_decay_list = []
    nu_mat = np.zeros([N_theta, N_theta])

    do_fit = True
    # do_fit = False

    inds_t_array = range(len(t_array))
    fig, ax = plt.subplots(1, 1,
                           figsize=(6, 6),
                           )
    # fig.suptitle(title)

    ## calculate the saturation value to estimate the rate
    # inds_t_saturation = range(7, 21)
    # inds_t_saturation = range(2, 3)
    inds_t_saturation = range(15, 31)  # for 2023 paper
    # inds_t_saturation = range(15, 29)

    # inds_t_saturation = range(len(t_array))

    N_curr = particles_counter_mat2_3d[0, 1, :]
    saturation_value = np.mean(N_curr[inds_t_saturation])
    label = '$\\bar{N}_{rc}$'
    label += '=' + '{:.3f}'.format(saturation_value)
    ax.plot(t_array, N_curr, color='b', linestyle='-', label=label)
    ax.hlines(saturation_value, t_array[inds_t_saturation[0]], t_array[inds_t_saturation[-1]],
              color='b', linewidth=2, linestyle='--')

    N_curr = particles_counter_mat2_3d[2, 1, :]
    saturation_value = np.mean(N_curr[inds_t_saturation])
    label = '$\\bar{N}_{lc}$'
    label += '=' + '{:.3f}'.format(saturation_value)
    ax.plot(t_array, N_curr, color='g', linestyle='-', label=label)
    ax.hlines(saturation_value, t_array[inds_t_saturation[0]], t_array[inds_t_saturation[-1]],
              color='g', linewidth=2, linestyle='--')

    N_curr = particles_counter_mat2_3d[1, 0, :]
    saturation_value = np.mean(N_curr[inds_t_saturation])
    label = '$\\bar{N}_{cr}$'
    label += '=' + '{:.3f}'.format(saturation_value)
    ax.plot(t_array, N_curr, color='r', linestyle='-', label=label)
    ax.hlines(saturation_value, t_array[inds_t_saturation[0]], t_array[inds_t_saturation[-1]],
              color='r', linewidth=2, linestyle='--')

    N_curr = particles_counter_mat2_3d[1, 2, :]
    saturation_value = np.mean(N_curr[inds_t_saturation])
    label = '$\\bar{N}_{cl}$'
    label += '=' + '{:.3f}'.format(saturation_value)
    ax.plot(t_array, N_curr, color='orange', linestyle='-', label=label)
    ax.hlines(saturation_value, t_array[inds_t_saturation[0]], t_array[inds_t_saturation[-1]],
              color='orange', linewidth=2, linestyle='--')

    LC_ini_fraction = np.sin(np.arcsin(field_dict['Rm'] ** (-0.5)) / 2) ** 2
    trapped_ini_fraction = 1 - 2 * LC_ini_fraction
    N_rc = particles_counter_mat2_3d[0, 1, :]
    N_cr = particles_counter_mat2_3d[1, 0, :]
    cone_escape_rate_1 = (N_rc * LC_ini_fraction - N_cr * trapped_ini_fraction) / LC_ini_fraction
    N_curr = cone_escape_rate_1
    # saturation_value = np.mean(N_curr[inds_t_saturation])
    label = '$(N_{rc}-N_{cr})/N_{cone}$'
    # label += '=' + '{:.3f}'.format(saturation_value)
    ax.plot(t_array, N_curr, color='black', linestyle='-', label=label)
    ax.hlines(saturation_value, t_array[inds_t_saturation[0]], t_array[inds_t_saturation[-1]],
              color='orange', linewidth=2, linestyle='--')

    # ax.plot(t_array, particles_counter_mat_3d[2, 1, :], color='g', linestyle='-', label='$\\bar{N}_{lc}$')
    # ax.plot(t_array, particles_counter_mat_3d[1, 0, :], color='r', linestyle='-', label='$\\bar{N}_{cr}$')
    # ax.plot(t_array, particles_counter_mat_3d[1, 2, :], color='orange', linestyle='-', label='$\\bar{N}_{cl}$')

    # ax.set_xlabel('$t \\cdot v_{th} / l$')
    # ax.set_xlabel('$t \\cdot v_{th,T} / l$')
    # ax.set_xlabel('$t / \\tau_{th}$',
    #               fontsize=20)
    # ax.set_ylim([0, 0.9])
    # ax.set_ylim([0, 1.0])
    # ax.set_xlim([0, 1.75])
    # ax.set_title(gas_name)
    # ax.set_title(title)
    # ax.legend(loc='upper left', fontsize=20)
    ax.legend(loc='upper left', fontsize=15)
    # ax.legend(loc='center right', fontsize=16)
    # ax.legend(fontsize=16)
    ax.grid(True)
    # text = '(a)'
    if gas_name == 'deuterium':
        gas_name_shorthand = 'D'
    if gas_name == 'tritium':
        gas_name_shorthand = 'T'
    # text = '(' + gas_name_shorthand + ',' + RF_set_name + ')'
    text = RF_set_name + ' (' + gas_name_shorthand + ')'
    plt.text(0.99, 0.98, text,
             # fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
             fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
             horizontalalignment='right', verticalalignment='top', color='k',
             transform=fig.axes[0].transAxes)
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
