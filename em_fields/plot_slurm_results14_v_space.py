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
# save_dir += '/set39_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set40_B0_1T_l_1m_Logan_Rm_3_intervals_D_T/'
save_dir += '/set42_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'

# RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
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

# alpha_loop_list = np.round(np.linspace(0.5, 1.5, 21), 2)  # set37, 39, 40
# beta_loop_list = np.round(np.linspace(-10, 10, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 21), 2)  # set38
# beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)


alpha_loop_list = np.round(np.linspace(0.5, 1.5, 7), 2)  # set42
beta_loop_list = np.round(np.linspace(-10, 10, 7), 2)


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

# ## For 2023 paper:
#
# select_alpha_list += [1.4]
# select_beta_list += [3.0]
# set_name_list += ['1']
#
# select_alpha_list += [1.0]
# select_beta_list += [-3.0]
# set_name_list += ['2']
#
# select_alpha_list += [0.7]
# select_beta_list += [-3.0]
# set_name_list += ['3']
#
# select_alpha_list += [0.55]
# select_beta_list += [-7.0]
# set_name_list += ['4']

## testing
induced_fields_factor = 1

# select_alpha_list += [1.0]
# select_beta_list += [0.0]
# set_name_list += ['1']

select_alpha_list += [1.5]
select_beta_list += [3.33]
set_name_list += ['2']

# for ind_set in range(4):
for ind_set in [0]:
    ind_set = 0
    # ind_set = 1
    # ind_set = 2
    # ind_set = 3

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
        set_name += '_iff' + str(induced_fields_factor)
        set_name += '_' + gas_name

        save_dir_curr = save_dir + set_name

        # load runs data
        data_dict_file = save_dir_curr + '.pickle'
        with open(data_dict_file, 'rb') as fid:
            data_dict = pickle.load(fid)
        settings_file = save_dir + 'settings.pickle'
        with open(settings_file, 'rb') as fid:
            settings = pickle.load(fid)
        # print('curr gas_name=', settings['gas_name'], ' v_th=', settings['v_th'])

        field_dict_file = save_dir + 'field_dict.pickle'
        with open(field_dict_file, 'rb') as fid:
            field_dict = pickle.load(fid)

        # for key in data_dict.keys():
        #     data_dict[key] = np.array(data_dict[key])

        # define v_th ref
        _, _, mi, _, Z_ion = define_plasma_parameters(gas_name='tritium')
        v_th_ref = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])

        # divide the phase space by the angle
        theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))

        # fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7), num=2)
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))

        ### plot the theretical resonance points
        _, _, mi, _, Z_ion = define_plasma_parameters(gas_name=gas_name)
        q = Z_ion * settings['e']  # Coulomb
        omega_cyc_0 = get_cyclotron_angular_frequency(q, field_dict['B0'], mi)
        omega_RF = alpha * field_dict['omega_cyclotron']
        omega_RF_over_omega_cyc_0 = omega_RF / omega_cyc_0
        k_RF = 2 * np.pi / field_dict['l'] * beta

        # calculate possible resonance points
        vz_arr = np.linspace(-3, 3, 400) * v_th_ref
        vt_arr = np.linspace(0, 3, 200) * v_th_ref
        resonance_possible_mat = np.zeros([len(vz_arr), len(vt_arr)])
        for ind_vz, vz_test in enumerate(vz_arr):
            for ind_vt, vt_test in enumerate(vt_arr):
                B0 = field_dict['B0']
                B_max_mirror = field_dict['B0'] * field_dict['Rm']
                # B_max_reversal = field_dict['B0'] * (1 + (vz_test / (vt_test + 1e-8 * v_th_ref)) ** 2.0)
                B_max_reversal = 1.25 * B0
                B_max = np.min([B_max_mirror, B_max_reversal])
                a = (omega_cyc_0 / B0) ** 2.0
                b = ((k_RF * vt_test) ** 2.0 - 2 * omega_RF * omega_cyc_0) / B0
                c = omega_RF ** 2.0 - k_RF ** 2.0 * (vz_test ** 2.0 + vt_test ** 2.0)
                determinant = b ** 2.0 - 4 * a * c
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

        # num_particles = 300
        # num_particles = 500
        num_particles = 1000
        # num_particles = 3000
        # colors = cm.rainbow(np.linspace(0, 1, num_particles))

        # dist_v_list = []
        # for ind_p in range(num_particles):
        #     v = data_dict['v'][ind_p, :]
        #     v0 = data_dict['v'][ind_p, 0]
        #     vt = data_dict['v_transverse'][ind_p, :]
        #     vt0 = data_dict['v_transverse'][ind_p, 0]
        #     vz = data_dict['v_axial'][ind_p, :]
        #     vz0 = data_dict['v_axial'][ind_p, 0]
        #     theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
        #     Bz = data_dict['Bz'][ind_p, :]
        #     Bz0 = data_dict['Bz'][ind_p, 0]
        #     vt_adjusted = vt * np.sqrt(
        #         Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)
        #
        #     # if (vt0 / v0) ** 2 < 1 / field_dict['Rm'] and vz0 > 0:
        #     #     vz_norm_loss_cone_r_list += [vz0 / settings['v_th']]
        #     # elif (vt0 / v0) ** 2 < 1 / field_dict['Rm'] and vz0 < 0:
        #     #     vz_norm_loss_cone_l_list += [vz0 / settings['v_th']]
        #     # else:
        #     #     vz_norm_trapped_list += [vz0 / settings['v_th']]
        #     #
        #     # if vz0 > 0:
        #     #     vz_pos_list +=[vz0 / settings['v_th']]
        #
        #     det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
        #     inds_positive = np.where(det > 0)[0]
        #     vz_adjusted = np.zeros(len(vz))
        #     vz_adjusted[inds_positive] = np.sign(vz0) * np.sqrt(det[inds_positive])
        #
        #     dist_v = max(np.sqrt((vz_adjusted - vz_adjusted[0]) ** 2 + (vt_adjusted - vt_adjusted[0]) ** 2))
        #     dist_v /= np.sqrt((vz_adjusted[0]) ** 2 + (vt_adjusted[0]) ** 2)
        #     dist_v_list += [dist_v]
        # max_dist_v = np.percentile(dist_v_list, 90)

        for ind_p in range(num_particles):

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
            dist_v /= (2 * v_th_ref)  # as in paper
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
        # file_name = 'v_space_set_' + gas_name_shorthand + '_' + RF_set_name
        # if RF_type == 'magnetic_transverse':
        #     file_name = 'BRF_' + file_name
        # # file_name += '_mod'
        # beingsaved = plt.gcf()
        # # beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
        # beingsaved.savefig(save_fig_dir + file_name + '.jpeg', format='jpeg', dpi=300)

### TODO: calculate the average vz in the loss cone
_, _, mi, _, Z_ion = define_plasma_parameters(gas_name=gas_name)
v_th = get_thermal_velocity(settings['T_keV'] * 1e3, mi, settings['kB_eV'])
print('v_th =', v_th)

# v0 = np.sqrt(v_0[:, 0] ** 2 + v_0[:, 1] ** 2 + v_0[:, 2] ** 2)
# vt0 = np.sqrt(v_0[:, 0] ** 2 + v_0[:, 1] ** 2)
# vz0 = v_0[:, 2]

v0 = data_dict['v'][:, 0]
vt0 = data_dict['v_transverse'][:, 0]
vz0 = data_dict['v_axial'][:, 0]

inds_right_loss_cone = [i for i in range(len(v0))
                        if vz0[i] > 0 and vt0[i] < vz0[i] * np.sqrt(1 / (field_dict['Rm'] - 1))]
print(np.mean(vz0[inds_right_loss_cone]) / v_th)

inds_left_loss_cone = [i for i in range(len(v0))
                       if vz0[i] < 0 and vt0[i] < abs(vz0[i]) * np.sqrt(1 / (field_dict['Rm'] - 1))]
print(np.mean(vz0[inds_left_loss_cone]) / v_th)

# according to https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution#Typical_speeds
# the integral of v*f(v) is:
v_MB_mean = 2 / np.sqrt(np.pi) * v_th
v_z_LC_average_theoretic = v_MB_mean / (2 * field_dict['Rm'] * (1 - np.sqrt(1 - 1 / field_dict['Rm'])))
print(v_z_LC_average_theoretic / v_th)

# # TODO: testing sphere randomization velocity
# # total_number_of_points = int(1e6)
# total_number_of_points = int(3e3)
# rand_unit_vec = np.random.randn(total_number_of_points, 3)
# for i in range(total_number_of_points):
#     rand_unit_vec[i, :] /= np.linalg.norm(rand_unit_vec[i, :])
# Rm = 3.0
# v0 = np.sqrt(rand_unit_vec[:, 0] ** 2 + rand_unit_vec[:, 1] ** 2 + rand_unit_vec[:, 2] ** 2)
# vt0 = np.sqrt(rand_unit_vec[:, 0] ** 2 + rand_unit_vec[:, 1] ** 2)
# vz0 = rand_unit_vec[:, 2]
# inds_cut_off = [i for i in range(len(v0))
#                         if vz0[i] > 0 and vt0[i] < vz0[i] * np.sqrt(1 / (Rm - 1))]
# vz0_unit_mean = np.mean(vz0[inds_cut_off])
# print('vz0_unit_mean = ', vz0_unit_mean)
# v_th = 1
# v_mean = 2 / np.sqrt(np.pi) * v_th
# print('v_mean = ', v_mean)
# vz_mean = v_mean * vz0_unit_mean
# print('vz_mean = ', vz_mean)
