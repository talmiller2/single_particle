import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel
from em_fields.default_settings import define_plasma_parameters
from em_fields.em_functions import get_thermal_velocity, get_cyclotron_angular_frequency

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

# figsize_large = (16, 9)
# figsize_large = (14, 7)

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
B_RF = 0.04  # T
# B_RF = 0.05  # T
# B_RF = 0.1  # T


gas_name_list = []
gas_name_list += ['deuterium']
# gas_name_list += ['DT_mix']
gas_name_list += ['tritium']

use_RF = True
# use_RF = False

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'
r_0 = 0
# r_0 = 1.5
# r_0 = 3.0

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
# beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.6, 1.0, 21), 2)  # set28
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
# beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 11), 2)  # set31, 32, 33
# beta_loop_list = np.round(np.linspace(-10, 0, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set34
# beta_loop_list = np.round(np.linspace(-5, 5, 11), 2)


# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 5), 2)  # set35
# beta_loop_list = np.round(np.linspace(-10, 0, 5), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.2, 21), 2)  # set36
# beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)

alpha_loop_list = np.round(np.linspace(0.5, 1.5, 21), 2)  # set37, 39
beta_loop_list = np.round(np.linspace(-10, 10, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 21), 2)  # set38
# beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)


select_alpha_list = []
select_beta_list = []
set_name_list = []

# select_alpha_list += [1.3]
# select_beta_list += [0.0]
# set_name_list += ['1']
#
# select_alpha_list += [1.4]
# select_beta_list += [3.0]
# set_name_list += ['2']
#
# select_alpha_list += [0.8]
# select_beta_list += [0.0]
# set_name_list += ['3']
#
# select_alpha_list += [1.0]
# select_beta_list += [-3.0]
# set_name_list += ['4']
#
# select_alpha_list += [0.6]
# select_beta_list += [-2.0]
# set_name_list += ['5']


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
# ind_set = 4

alpha = select_alpha_list[ind_set]
beta = select_beta_list[ind_set]
RF_set_name = set_name_list[ind_set]

for gas_name in gas_name_list:

    if use_RF is False:
        title = 'without RF'
    elif RF_type == 'electric_transverse':
        title = '$E_{RF}$=' + str(E_RF_kVm) + 'kV/m, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
    elif RF_type == 'magnetic_transverse':
        title = '$B_{RF}$=' + str(B_RF) + 'T, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
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
    if absolute_velocity_sampling_type == 'const_vth':
        set_name = 'const_vth_' + set_name
    if r_0 > 0:
        set_name += '_r0_' + str(r_0) + '_' + set_name
    # set_name += '_antiresonant'
    set_name += '_' + gas_name

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

    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key])

    # define v_th ref
    _, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
    v_th_ref = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])

    # divide the phase space by the angle

    theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))

    # number_of_time_intervals = 3
    number_of_time_intervals = data_dict['t'].shape[1]

    from matplotlib import cm

    # colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
    # # colors = ['b', 'g', 'r']
    # # colors = ['r', 'g', 'b']
    #
    # for ind_t in range(number_of_time_intervals):
    #     # for ind_t in [0, 10]:
    #     # for ind_t in [0, 1]:
    #     # for ind_t in [0, 10, 20]:
    #     #     print(ind_t)
    #
    #     # inds_particles = range(data_dict['t'].shape[0])
    #     # inds_particles = [0, 1, 2]
    #     # inds_particles = range(1001)
    #     inds_particles = range(100)
    #
    #     if ind_t == 0:
    #         print('num particles = ' + str(len(inds_particles)))
    #
    #     v = data_dict['v'][inds_particles, ind_t]
    #     v0 = data_dict['v'][inds_particles, 0]
    #     vt = data_dict['v_transverse'][inds_particles, ind_t]
    #     vt0 = data_dict['v_transverse'][inds_particles, 0]
    #     vz = data_dict['v_axial'][inds_particles, ind_t]
    #     vz0 = data_dict['v_axial'][inds_particles, 0]
    #     theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
    #     Bz = data_dict['Bz'][inds_particles, ind_t]
    #     Bz0 = data_dict['Bz'][inds_particles, 0]
    #     vt_adjusted = vt * np.sqrt(Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)
    #
    #     # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt0 ** 2.0 * (Bz / Bz0 - 1))
    #     # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz))
    #     # theta_adjusted = np.mod(360 / (2 * np.pi) * np.arctan(vt_adjusted / vz_adjusted), 180)
    #
    #     det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
    #     inds_positive = np.where(det > 0)[0]
    #     vz_adjusted = np.zeros(len(inds_particles))
    #     vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])
    #     # vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(det[inds_positive])
    #
    #     theta_adjusted = 90.0 * np.ones(len(inds_particles))
    #     theta_adjusted[inds_positive] = np.mod(
    #         360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)
    #
    #     color = colors[ind_t]
    #
    #     if ind_t == 0:
    #         fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7), num=1)
    #     if np.mod(ind_t, 5) == 1:
    #         label = str(ind_t)
    #         label = '$t \\cdot v_{th} / l$=' + '{:.2f}'.format(
    #             data_dict['t'][0, ind_t] / (settings['l'] / settings['v_th']))
    #         ax1.scatter(vz_adjusted / settings['v_th'], vt_adjusted / settings['v_th'], color=color, alpha=0.2, label=label)
    #         if ind_t == 0:
    #             # plot the diagonal LC lines
    #             vz_axis = np.array([0, 2 * settings['v_th']])
    #             vt_axis = vz_axis * np.sqrt(1 / (field_dict['Rm'] - 1))
    #             ax1.plot(vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
    #             ax1.plot(-vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
    # ax1.set_xlabel('$v_z / v_{th}$')
    # ax1.set_ylabel('$v_{\\perp} / v_{th}$')
    # ax1.legend()
    # ax1.grid(True)

    # fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7), num=2)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))

    # # # TODO: testing
    # # alpha = 1.3
    # alpha = 1.2
    # # beta = 5.0

    ### plot the theretical resonance points
    _, _, mi, _, Z_ion = define_plasma_parameters(gas_name=gas_name)
    q = Z_ion * settings['e']  # Coulomb
    omega_cyc_0 = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)
    omega_RF = alpha * field_dict['omega_cyclotron']
    omega_RF_over_omega_cyc_0 = omega_RF / omega_cyc_0
    k_RF = 2 * np.pi / field_dict['l'] * beta
    v_RF = omega_RF / k_RF
    # v_RF /= v_th_ref # normalized velocity

    # calculate possible resonance points
    vz_arr = np.linspace(-3, 3, 400) * v_th_ref
    vt_arr = np.linspace(0, 3, 200) * v_th_ref
    resonance_possible_mat = np.zeros([len(vz_arr), len(vt_arr)])
    for ind_vz, vz_test in enumerate(vz_arr):
        for ind_vt, vt_test in enumerate(vt_arr):
            B0 = field_dict['B0']
            B_max = field_dict['B0'] * field_dict['Rm']
            B_res = field_dict['B0']
            # a = (1 / (B0 * omega_RF_over_omega_cyc_0)) ** 2 + (vt_test / (v_RF * B_res)) ** 2
            # b = - 2 / (B0 * omega_RF_over_omega_cyc_0)
            # c = 1 - (vt_test ** 2 + vz_test ** 2) / v_RF ** 2
            a = (omega_cyc_0 / B0) ** 2
            b = ((k_RF * vt_test) ** 2 - 2 * omega_RF * omega_cyc_0) / B0
            c = - k_RF ** 2 * (vz_test ** 2 + vt_test ** 2)
            determinant = b ** 2 - 4 * a * c
            if determinant >= 0:
                B_sol1 = (- b + np.sqrt(determinant)) / (2 * a)
                B_sol2 = (- b - np.sqrt(determinant)) / (2 * a)
                # if B_sol1 >= B0 and B_sol1 <= B_max and B_sol2 >= B0 and B_sol2 <= B_max:
                #     print('found')
                resonant_solution_found = False
                for B_sol in [B_sol1, B_sol2]:
                    if B_sol >= B0 and B_sol <= B_max and resonant_solution_found == False:
                        # vz_B_sol = v_RF * (1 - B_sol / B0 / omega_RF_over_omega_cyc_0)
                        vz_B_sol = (omega_RF - omega_cyc_0 * B_sol / B0) / k_RF

                        # if np.sign(vz_B_sol) == np.sign(vz_test):
                        #     resonance_possible_mat[ind_vz, ind_vt] = 1
                        #     resonant_solution_found = True

                        # we care about the axial direction only for particles that are in the loss cone initially
                        if vt_test > abs(vz_test * np.sqrt(1 / (field_dict['Rm'] - 1))):  # outside of loss-cones
                            resonance_possible_mat[ind_vz, ind_vt] = 1
                            resonant_solution_found = True
                        else:
                            if np.sign(vz_B_sol) == np.sign(vz_test):
                                resonance_possible_mat[ind_vz, ind_vt] = 1
                                resonant_solution_found = True

    # # mirror the resonance points outside of the loss cones (vz,vt)=(-vz,vt)
    # for ind_vz, vz_test in enumerate(vz_arr):
    #     for ind_vt, vt_test in enumerate(vt_arr):
    #         ind_vz_mirrored =  len(vz_arr) - ind_vz - 1
    #         if vt_test > abs(vz_test * np.sqrt(1 / (field_dict['Rm'] - 1))): # outside of loss-cones
    #             if resonance_possible_mat[ind_vz, ind_vt] == 1:
    #                 resonance_possible_mat[ind_vz_mirrored, ind_vt] = 1

    # find the min-max resonance vz, for each vt separately
    vt_min_array = np.zeros(len(vz_arr))
    vt_max_array = np.zeros(len(vz_arr))
    for ind_vz in range(len(vz_arr)):
        if len(np.where(resonance_possible_mat[ind_vz, :] == 1)[0]) > 0:
            vt_min_array[ind_vz] = vt_arr[np.where(resonance_possible_mat[ind_vz, :] == 1)[0][0]]
            vt_max_array[ind_vz] = vt_arr[np.where(resonance_possible_mat[ind_vz, :] == 1)[0][-1]]
        else:
            vt_min_array[ind_vz] = np.nan
            vt_max_array[ind_vz] = np.nan
    vt_min_array /= v_th_ref  # normalized velocity
    vt_max_array /= v_th_ref  # normalized velocity
    vz_arr /= v_th_ref  # normalized velocity
    vt_arr /= v_th_ref  # normalized velocity

    ax2.fill_between(vz_arr, vt_min_array, vt_max_array,
                     color='grey', alpha=0.25,
                     # color='pink',
                     )

    # fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
    # ax3.imshow(resonance_possible_mat)

    ### plot particle trajectories

    # num_particles = 100
    num_particles = 1000
    # num_particles = 3000
    # colors = cm.rainbow(np.linspace(0, 1, num_particles))

    dist_v_list = []
    for ind_p in range(num_particles):
        v = data_dict['v'][ind_p, :]
        v0 = data_dict['v'][ind_p, 0]
        vt = data_dict['v_transverse'][ind_p, :]
        vt0 = data_dict['v_transverse'][ind_p, 0]
        vz = data_dict['v_axial'][ind_p, :]
        vz0 = data_dict['v_axial'][ind_p, 0]
        theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
        Bz = data_dict['Bz'][ind_p, :]
        Bz0 = data_dict['Bz'][ind_p, 0]
        vt_adjusted = vt * np.sqrt(
            Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

        # if (vt0 / v0) ** 2 < 1 / field_dict['Rm'] and vz0 > 0:
        #     vz_norm_loss_cone_r_list += [vz0 / settings['v_th']]
        # elif (vt0 / v0) ** 2 < 1 / field_dict['Rm'] and vz0 < 0:
        #     vz_norm_loss_cone_l_list += [vz0 / settings['v_th']]
        # else:
        #     vz_norm_trapped_list += [vz0 / settings['v_th']]
        #
        # if vz0 > 0:
        #     vz_pos_list +=[vz0 / settings['v_th']]

        det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
        inds_positive = np.where(det > 0)[0]
        vz_adjusted = np.zeros(len(vz))
        vz_adjusted[inds_positive] = np.sign(vz0) * np.sqrt(det[inds_positive])

        dist_v = max(np.sqrt((vz_adjusted - vz_adjusted[0]) ** 2 + (vt_adjusted - vt_adjusted[0]) ** 2))
        dist_v /= np.sqrt((vz_adjusted[0]) ** 2 + (vt_adjusted[0]) ** 2)
        dist_v_list += [dist_v]
    max_dist_v = np.percentile(dist_v_list, 90)

    for ind_p in range(num_particles):

        v = data_dict['v'][ind_p, :]
        v0 = data_dict['v'][ind_p, 0]
        vt = data_dict['v_transverse'][ind_p, :]
        vt0 = data_dict['v_transverse'][ind_p, 0]
        vz = data_dict['v_axial'][ind_p, :]
        vz0 = data_dict['v_axial'][ind_p, 0]
        theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
        Bz = data_dict['Bz'][ind_p, :]
        Bz0 = data_dict['Bz'][ind_p, 0]
        vt_adjusted = vt * np.sqrt(
            Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

        # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt0 ** 2.0 * (Bz / Bz0 - 1))
        # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz))
        # theta_adjusted = np.mod(360 / (2 * np.pi) * np.arctan(vt_adjusted / vz_adjusted), 180)

        det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
        inds_positive = np.where(det > 0)[0]
        vz_adjusted = np.zeros(len(vz))
        vz_adjusted[inds_positive] = np.sign(vz0) * np.sqrt(det[inds_positive])
        # vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(det[inds_positive])

        # theta_adjusted = 90.0 * np.ones(len(inds_particles))
        # theta_adjusted[inds_positive] = np.mod(
        # 360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)

        dist_v = max(np.sqrt((vz_adjusted - vz_adjusted[0]) ** 2 + (vt_adjusted - vt_adjusted[0]) ** 2))
        # dist_v /= np.sqrt((vz_adjusted[0]) ** 2 + (vt_adjusted[0]) ** 2)
        # dist_v /= settings['v_th_for_cyc']
        # dist_v *= 1.2
        dist_v /= v_th_ref
        # dist_v *= 1.2
        dist_v *= 1.0
        # dist_v /= 2
        dist_v /= max_dist_v
        color = cm.rainbow(dist_v)

        ax2.plot(vz_adjusted / v_th_ref, vt_adjusted / v_th_ref,
                 # color=colors[ind_p],
                 color=color,
                 alpha=0.2,
                 )
        ax2.plot(vz_adjusted[0] / v_th_ref, vt_adjusted[0] / v_th_ref,
                 # color=colors[ind_p],
                 color=color,
                 marker='o',
                 )
        # ax2.plot(vz_adjusted[-1] / settings['v_th'], vt_adjusted[-1] / settings['v_th'],
        #          color=colors[ind_p], marker='o', fillstyle='none',
        #          )

        # plot the diagonal LC lines
        # if ind_p == 0:
        if ind_p == num_particles - 1:
            vz_axis = np.array([0, 4 * v_th_ref])
            vt_axis = vz_axis * np.sqrt(1 / (field_dict['Rm'] - 1))
            ax2.plot(vz_axis / v_th_ref, vt_axis / v_th_ref, color='k', linestyle='--')
            ax2.plot(-vz_axis / v_th_ref, vt_axis / v_th_ref, color='k', linestyle='--')

    # text = '(' + RF_set_name + ')'
    if gas_name == 'deuterium':
        gas_name_shorthand = 'D'
    if gas_name == 'tritium':
        gas_name_shorthand = 'T'
    text = RF_set_name + ' (' + gas_name_shorthand + ')'
    # text = '(' + gas_name_shorthand + ',' + RF_set_name + ')'
    # text = '(b)'
    ax2.text(0.20, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
             horizontalalignment='right', verticalalignment='top',
             transform=fig2.axes[0].transAxes)

    # ax2.set_xlabel('$v_z / v_{th}$')
    # ax2.set_ylabel('$v_{\\perp} / v_{th}$')
    if ind_set == 3:
        ax2.set_xlabel('$v_z / v_{th,T}$', fontsize=20)
    if gas_name == 'deuterium':
        ax2.set_ylabel('$v_{\\perp} / v_{th,T}$', fontsize=20)
    # ax2.set_title(title)
    # ax2.set_xlim([-2.0, 2.0])
    ax2.set_xlim([-2.5, 2.5])
    # ax2.set_ylim([0, 2.0])
    ax2.set_ylim([0, 2.5])
    # ax2.set_ylim([0, 3.0])
    # fig2.set_tight_layout(True)
    fig2.set_layout_engine(layout='tight')
    # ax2.legend()
    ax2.grid(True)

    # ## save plots to file
    # save_fig_dir = '../../../Papers/texts/paper2022/pics/'
    # # file_name = 'v_space_evolution_' + set_name
    # # file_name = 'v_space_evolution_set_' + set_num
    # file_name = 'v_space_set_' + gas_name_shorthand + '_' + RF_set_name
    # if RF_type == 'magnetic_transverse':
    #     file_name += '_BRF'
    # beingsaved = plt.gcf()
    # # # beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
    # beingsaved.savefig(save_fig_dir + file_name + '.jpeg', format='jpeg', dpi=300)
