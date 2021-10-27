import pickle

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

plt.close('all')

save_dir_main = '/Users/talmiller/Downloads/single_particle/'
# save_dir_main += '/set4/'
# save_dir_main += '/set5/'
# save_dir_main += '/set7_T_10keV_B0_1T_Rm_2_l_1m/'
save_dir_main += '/set8_T_10keV_B0_1T_Rm_2_l_1m/'

set_names = []

# ERF = 0
ERF = 10
# ERF = 30

# alpha = 0.6
# alpha = 0.8
# alpha = 1.0
# alpha = 1.2
alpha = 1.5
# alpha = 2.0
# alpha = 3.0

# vz_res = 0.5
# vz_res = 1.0
# vz_res = 1.5
vz_res = 2.0
# vz_res = 2.5
# vz_res = 3.0

if ERF > 0:
    set_names += ['ERF_' + str(ERF) + '_alpha_' + str(alpha) + '_vz_' + str(vz_res)]
else:
    set_names += ['ERF_0']

for set_ind in range(len(set_names)):
    set_name = set_names[set_ind]
    save_dir = save_dir_main + set_name

    # load runs data
    data_dict_file = save_dir + '.pickle'
    with open(data_dict_file, 'rb') as fid:
        data_dict = pickle.load(fid)
    settings = data_dict['settings']
    field_dict = data_dict['field_dict']

    # draw trajectories for several particles
    num_particles = len(data_dict['z'])
    # ind_points = [0, 1, 2, 4, 5]
    # ind_points = [0]
    # ind_points = [3]
    # ind_points = range(5)
    # ind_points = range(10)
    # ind_points = range(20)
    # ind_points = range(100)
    # ind_points = range(300)
    ind_points = range(1000)
    # ind_points = range(100, 200)
    # ind_points = range(20, 30)
    # ind_points = range(30, 40)
    # ind_points = range(num_particles)

    # do_particles_plot = False
    do_particles_plot = True

    if do_particles_plot:
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
        # fig = plt.figure(1, figsize=(16, 6))
        # # fig = plt.figure(1)
        # if fig.axes == []:
        #     ax1 = plt.subplot(1, 4, 1)
        #     ax2 = plt.subplot(1, 4, 2)
        #     ax3 = plt.subplot(1, 4, 3)
        #     ax4 = plt.subplot(1, 4, 4)
        # else:
        #     [ax1, ax2, ax3, ax4] = fig.axes

        # fig = plt.figure(2, figsize=(12, 5))
        # if fig.axes == []:
        #     ax1 = plt.subplot(1, 2, 1)
        #     ax2 = plt.subplot(1, 2, 2)
        # else:
        #     [ax1, ax2] = fig.axes

        pass

    # for ind_point in ind_points:
    for ind_point in ind_points:
        # skip = 1
        # # skip = 2
        # # num_snapshots = len(data_dict['t'][ind_point])
        # t = np.array(data_dict['t'][ind_point])[0::skip]
        # z = np.array(data_dict['z'][ind_point])[0::skip]
        # v = np.array(data_dict['v'][ind_point])[0::skip]
        # v_transverse = np.array(data_dict['v_transverse'][ind_point])[0::skip]
        # v_axial = np.array(data_dict['v_axial'][ind_point])[0::skip]
        # Bz = np.array(data_dict['Bz'][ind_point])[0::skip]

        t = np.array(data_dict['t'][ind_point])
        z = np.array(data_dict['z'][ind_point])
        v = np.array(data_dict['v'][ind_point])
        v_transverse = np.array(data_dict['v_transverse'][ind_point])
        v_axial = np.array(data_dict['v_axial'][ind_point])
        Bz = np.array(data_dict['Bz'][ind_point])

        if len(data_dict['t'][ind_point]) > 0:

            # pick only the indices where v_axial is the same direction as the initial v_axial
            vz = np.array(data_dict['v_axial'][ind_point])
            # # positive_z_velocity = np.sign(vz[0])
            # positive_z_velocity = np.sign(data_dict['v_0'][ind_point, 2])
            # inds_in_trajectory = np.where(vz * positive_z_velocity > 0)[0]
            # t = np.array(data_dict['t'][ind_point])[inds_in_trajectory]
            # z = np.array(data_dict['z'][ind_point])[inds_in_trajectory]
            # v = np.array(data_dict['v'][ind_point])[inds_in_trajectory]
            # v_transverse = np.array(data_dict['v_transverse'][ind_point])[inds_in_trajectory]
            # v_axial = np.array(data_dict['v_axial'][ind_point])[inds_in_trajectory]
            # Bz = np.array(data_dict['Bz'][ind_point])[inds_in_trajectory]

            # calculate if a particle is initially in right loss cone
            # LC_cutoff = field_dict['Rm'] ** (-0.5)
            # in_loss_cone = v_transverse[0] / v[0] < LC_cutoff
            in_loss_cone = (v_transverse[0] / v[0]) ** 2 < 1 / field_dict['Rm']
            # in_loss_cone = (v_transverse[0] / v_axial[0]) ** 2 < 1 / (field_dict['Rm'] - 1.0)
            positive_z_velocity = v_axial[0] > 0

            if in_loss_cone and positive_z_velocity:  # right loss cone
                linestyle = '-'
                linewidth = 1
                do_plot = True
                # do_plot = False
            elif in_loss_cone and not positive_z_velocity:  # left loss cone
                # linestyle = ':'
                linestyle = '-'
                linewidth = 1
                do_plot = True
                # do_plot = False
            else:  # trapped
                # linestyle = '--'
                linestyle = '-'
                linewidth = 1
                do_plot = True
                # do_plot = False

            # if not positive_z_velocity:  # draw points that started with negative velocity, on negative side of the plot
            #     v_axial *= -1

            # plots
            if do_plot and do_particles_plot:
                # ax1.plot(t_array / field_dict['tau_cyclotron'], z / settings['l'], label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth)
                # ax1.set_xlabel('$t/\\tau_{cyc}$')
                # ax1.set_ylabel('$z/l$')
                #
                # ax2.plot(t_array / field_dict['tau_cyclotron'], v / settings['v_th'], label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth)
                # ax2.set_xlabel('$t/\\tau_{cyc}$')
                # ax2.set_ylabel('$|v|/v_{th}$')
                #
                # ax3.plot(t_array / field_dict['tau_cyclotron'], v_transverse / settings['v_th'], label=ind_point,
                #          linestyle=linestyle, linewidth=linewidth)
                # ax3.set_xlabel('$t/\\tau_{cyc}$')
                # ax3.set_ylabel('$v_{\perp}/v_{th}$')
                #
                # # ax4.plot(v_axial / settings['v_th'], label=ind_point, linestyle=linestyle, linewidth=linewidth)
                # ax4.plot(t_array / field_dict['tau_cyclotron'], v_transverse / v, label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth)
                # ax4.set_xlabel('$t/\\tau_{cyc}$')
                # # ax4.set_ylabel('$v_{z}/v_{th}$')
                # ax4.set_ylabel('$v_{\perp}/|v|$')

                # plt.figure(2)
                # E = (v / settings['v_th']) ** 2
                # E_transverse = (v_transverse / settings['v_th']) ** 2
                # plt.plot(t_array / field_dict['tau_cyclotron'], E_transverse, label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth)
                #
                # plt.figure(3)
                # plt.plot(t_array / field_dict['tau_cyclotron'], z / settings['l'], label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth, marker='o', markersize=2,)

                plt.figure(4)
                plt.plot(v_axial / settings['v_th'], v_transverse / settings['v_th'], label=ind_point,
                         linewidth=linewidth,
                         # linestyle=linestyle,
                         linestyle='none', marker='o', markersize=2,
                         )
                plt.plot(v_axial[0] / settings['v_th'], v_transverse[0] / settings['v_th'], 'ko', markersize=2)

                # plt.figure(5)
                # plt.plot(t / field_dict['tau_cyclotron'], Bz, '-o', label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth)

                # E = (v / settings['v_th']) ** 2
                # E_transverse = (v_transverse / settings['v_th']) ** 2
                # ax1.plot(t_array / field_dict['tau_cyclotron'], E_transverse, label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth)
                #
                # ax2.plot(t_array / field_dict['tau_cyclotron'], z / settings['l'], label=ind_point, linestyle=linestyle,
                #          linewidth=linewidth)

    if do_particles_plot:
        # ax1.grid(True)
        # ax2.grid(True)
        # ax3.grid(True)
        # ax4.grid(True)
        #
        # ax4.plot(t_array / field_dict['tau_cyclotron'], LC_cutoff * np.ones(len(t_array)), '-k', label='LC cutoff',
        #          linewidth=3)
        # # ax4.legend()

        # plt.figure(2)
        # plt.grid(True)
        # plt.xlabel('$t/\\tau_{cyc}$')
        # plt.ylabel('$E_{\perp}/E_{th}$')
        # plt.tight_layout()
        #
        # plt.figure(3)
        # plt.grid(True)
        # plt.xlabel('$t/\\tau_{cyc}$')
        # plt.ylabel('$z/l$')
        # plt.tight_layout()

        # check for all initial trajectories, if they can resonate down the mirror

        # vz_test = 1.5
        # vt_test = 0.5
        # v_tilde = vz_res / (alpha - 1.0)
        # determinant =  4 * alpha ** 2 * v_tilde ** 4 - 4 * (v_tilde ** 2 + vt_test ** 2) * (alpha ** 2 * v_tilde ** 2 - vz_test ** 2 - vz_test ** 2)
        # B_sol1 = ( 2 * alpha + v_tilde ** 2 + np.sqrt(determinant) ) / (2 * (v_tilde ** 2 + vt_test ** 2))
        # B_sol2 = ( 2 * alpha - v_tilde ** 2 + np.sqrt(determinant) ) / (2 * (v_tilde ** 2 + vt_test ** 2))
        #
        # if (B_sol1 >= 1 and B_sol1 <= field_dict['Rm']) or (B_sol2 >= 1 and B_sol2 <= field_dict['Rm']):
        #     left_going_resonates = True
        # else:
        #     left_going_resonates = False

        # vz_arr = - np.linspace(0, 4, 40)
        # vt_arr = np.linspace(0, 4, 40)

        vz_arr = - np.sign(alpha - 1) * np.linspace(0, 3, 100)
        vt_arr = np.linspace(0, 3, 100)

        # vz_valid = []
        # vt_valid = []
        # for vz_test in vz_arr:
        #     for vt_test in vt_arr:
        #         left_going_resonates = False
        #
        #         v_tilde = vz_res / (alpha - 1.0)
        #         determinant = 4 * alpha ** 2 * v_tilde ** 4 - 4 * (v_tilde ** 2 + vt_test ** 2) * (
        #                     alpha ** 2 * v_tilde ** 2 - vz_test ** 2 - vz_test ** 2)
        #         if determinant >= 0:
        #             B_sol1 = (2 * alpha * v_tilde ** 2 + np.sqrt(determinant)) / (2 * (v_tilde ** 2 + vt_test ** 2))
        #             B_sol2 = (2 * alpha * v_tilde ** 2 - np.sqrt(determinant)) / (2 * (v_tilde ** 2 + vt_test ** 2))
        #             if (B_sol1 >= 1 and B_sol1 <= field_dict['Rm']) or (B_sol2 >= 1 and B_sol2 <= field_dict['Rm']):
        #             # if1 (B_sol1 >= 1 and B_sol1 <= field_dict['Rm']):
        #             # if (B_sol2 >= 1 and B_sol2 <= field_dict['Rm']):
        #                 left_going_resonates = True
        #             # if B_sol1 < 0 or B_sol2 < 0:
        #             #     print('vz=' + str(vz_test) + ', vt=' + str(vt_test) + ': B_sols=' + str(B_sol1) + ',' + str(B_sol2))
        #
        #         if left_going_resonates:
        #             vz_valid += [vz_test]
        #             vt_valid += [vt_test]

        vt_min_array = np.zeros(len(vz_arr))
        vt_max_array = np.zeros(len(vz_arr))
        for ind_vz, vz_test in enumerate(vz_arr):
            vt_valid_points = []
            for vt_test in vt_arr:
                left_going_resonates = False

                v_tilde = vz_res / (alpha - 1.0)
                determinant = 4 * alpha ** 2 * v_tilde ** 4 - 4 * (v_tilde ** 2 + vt_test ** 2) * (
                        alpha ** 2 * v_tilde ** 2 - vz_test ** 2 - vz_test ** 2)
                if determinant >= 0:
                    B_sol1 = (2 * alpha * v_tilde ** 2 + np.sqrt(determinant)) / (2 * (v_tilde ** 2 + vt_test ** 2))
                    B_sol2 = (2 * alpha * v_tilde ** 2 - np.sqrt(determinant)) / (2 * (v_tilde ** 2 + vt_test ** 2))
                    if (B_sol1 >= 1 and B_sol1 <= field_dict['Rm']) or (B_sol2 >= 1 and B_sol2 <= field_dict['Rm']):
                        # if (B_sol1 >= 1) or (B_sol2 >= 1):
                        # if1 (B_sol1 >= 1 and B_sol1 <= field_dict['Rm']):
                        # if (B_sol2 >= 1 and B_sol2 <= field_dict['Rm']):
                        left_going_resonates = True
                    # if B_sol1 < 0 or B_sol2 < 0:
                    #     print('vz=' + str(vz_test) + ', vt=' + str(vt_test) + ': B_sols=' + str(B_sol1) + ',' + str(B_sol2))
                    # if B_sol1 > 0 and B_sol2 > 0:
                    #     print('vz=' + str(vz_test) + ', vt=' + str(vt_test) + ': B_sols=' + str(B_sol1) + ',' + str(B_sol2))

                if left_going_resonates:
                    vt_valid_points += [vt_test]

            if len(vt_valid_points) > 0:
                vt_min_array[ind_vz] = np.min(vt_valid_points)
                vt_max_array[ind_vz] = np.max(vt_valid_points)
            else:
                vt_min_array[ind_vz] = np.nan
                vt_max_array[ind_vz] = np.nan

        plt.figure(4)
        v_arr = np.linspace(0, 3, 100)
        plt.plot(v_arr, np.sqrt(1 / (field_dict['Rm'] - 1.0)) * v_arr, '-k', linewidth=3)
        plt.plot(-v_arr, np.sqrt(1 / (field_dict['Rm'] - 1.0)) * v_arr, '-k', linewidth=3)
        # plt.plot(vz_res + 0 * v_arr, v_arr, '--k', linewidth=3)
        # plt.plot(v_arr, np.sqrt((v_arr ** 2.0 - vz_res ** 2.0) / (field_dict['Rm'] ** 2.0 - 1.0)) , '--k', linewidth=3)
        # plt.plot(-vz_res + 0 * v_arr, v_arr, '--k', linewidth=3)
        # plt.plot(-v_arr, np.sqrt((v_arr ** 2.0 - vz_res ** 2.0) / (field_dict['Rm'] ** 2.0 - 1.0)), '--k', linewidth=3)
        # plt.scatter(vz_valid, vt_valid, color='m', s=2)
        # plt.scatter(vz_res + 0 * vt_arr, vt_arr, color='m', s=2)
        plt.fill_between(vz_arr, vt_min_array, vt_max_array, color='m', alpha=0.5)
        plt.plot(vz_res + 0 * vt_arr, vt_arr, color='m', linewidth=3, alpha=0.5)
        plt.grid(True)
        plt.xlabel('$v_{\parallel}/v_{th}$')
        plt.ylabel('$v_{\perp}/v_{th}$')
        plt.tight_layout()

        # plt.figure(5)
        # plt.grid(True)
        # plt.xlabel('$t/\\tau_{cyc}$')
        # plt.ylabel('$B_z$')
        # plt.tight_layout()

        # ax1.grid(True)
        # ax1.set_xlabel('$t/\\tau_{cyc}$')
        # ax1.set_ylabel('$E_{\perp}/E_{th}$')
        # ax2.grid(True)
        # ax2.set_xlabel('$t/\\tau_{cyc}$')
        # ax2.set_ylabel('$z/l$')
        # fig.tight_layout()
