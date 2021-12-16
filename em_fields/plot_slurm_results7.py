import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# plt.close('all')

save_dir_main = '/Users/talmiller/Downloads/single_particle/'
# save_dir_main += '/set14_T_B0_1T_l_1m_randphase_save_intervals/'
# save_dir_main += '/set15_T_B0_1T_l_1m_Logan_intervals/'
# save_dir_main += '/set16_T_B0_1T_l_1m_Post_intervals/'
# save_dir_main += '/set17_T_B0_1T_l_3m_Post_intervals/'
# save_dir_main += '/set18_T_B0_1T_l_3m_Logan_intervals/'
# save_dir_main += '/set19_T_B0_1T_l_3m_Post_intervals_Rm_1.3/'
# save_dir_main += '/set20_B0_1T_l_3m_Post_intervals_Rm_3/'
save_dir_main += '/set21_B0_1T_l_3m_Post_intervals_Rm_3_different_phases/'

set_names = []

# Rm = 1.3
# Rm = 2
Rm = 3
# Rm = 4

# ERF = 0
# ERF = 1
# ERF = 5
# ERF = 10
# ERF = 30
ERF = 100

# alpha = 0.6
# alpha = 0.8
# alpha = 1.0
# alpha = 1.2
alpha = 1.5
# alpha = 2.0
# alpha = 2.5
# alpha = 3.0

# vz_res = 0.5
vz_res = 1.0
# vz_res = 1.5
# vz_res = 2.0
# vz_res = 2.5
# vz_res = 3.0

# color = 'b'
# color = 'g'
# color = 'r'
# color = 'm'

omega_RF_over_omega_cyc_0 = alpha
# if alpha == 1.0:
#     v_RF_label = '$\\infty$'
# else:
#     v_RF = vz_res * alpha / (alpha - 1.0)
#     v_RF_label = '{:.2f}'.format(v_RF)

alpha_actual = alpha * (1 + np.pi / 100)
v_RF = vz_res * alpha_actual / (alpha_actual - 1.0)
v_RF_label = '{:.2f}'.format(v_RF)
alpha_actual_label = '{:.2f}'.format(alpha_actual)

print('vz_res/v_th = ' + str(vz_res) + ', alpha = ' + str(alpha))
print('omega_RF/omega_cyc0 = ' + '{:.2f}'.format(omega_RF_over_omega_cyc_0) + ', v_RF/v_th = ' + v_RF_label)

if ERF > 0:
    set_names += ['Rm_' + str(int(Rm)) + '_ERF_' + str(ERF) + '_alpha_' + str(alpha) + '_vz_' + str(vz_res)]
else:
    set_names += ['Rm_' + str(int(Rm)) + '_ERF_0']

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
    # ind_points = [801]
    # ind_points = [537, 752, 802]
    # ind_points = [1, 4]
    # ind_points = range(5)
    # ind_points = range(10)
    # ind_points = range(20)
    # ind_points = range(100)
    # ind_points = range(500)
    ind_points = range(1000)
    # ind_points = range(2000)
    # ind_points = range(100, 200)
    # ind_points = range(20, 30)
    # ind_points = range(30, 40)
    # ind_points = range(num_particles)

    points_counter = 0
    points_right_counter = 0
    points_left_counter = 0
    points_trapped_counter = 0

    for ind_point in ind_points:

        inds_trajectory = range(len(data_dict['t'][ind_point]))
        # inds_trajectory = np.argsort(data_dict['t'][ind_point])

        t = np.array(data_dict['t'][ind_point])[inds_trajectory]
        z = np.array(data_dict['z'][ind_point])[inds_trajectory]
        v = np.array(data_dict['v'][ind_point])[inds_trajectory]
        v_transverse = np.array(data_dict['v_transverse'][ind_point])[inds_trajectory]
        v_axial = np.array(data_dict['v_axial'][ind_point])[inds_trajectory]
        Bz = np.array(data_dict['Bz'][ind_point])[inds_trajectory]

        if ind_point == 0:
            counter_right_particles = 0 * t
            counter_right_particles2 = 0 * t
            counter_left_particles = 0 * t
            counter_left_particles2 = 0 * t
            counter_trapped_particles = 0 * t
            counter_trapped_particles2 = 0 * t
            counter_trapped_particles3 = 0 * t
            counter_trapped_particles4 = 0 * t

            counter_right_trapped_first_cell = 0
            counter_left_trapped_first_cell = 0
            counter_trapped_escaped_right_first_cell = 0
            counter_trapped_escaped_left_first_cell = 0

        # calculate if a particle is initially in right loss cone
        in_loss_cone = (v_transverse[0] / v[0]) ** 2 < 1 / field_dict['Rm']
        positive_z_velocity = v_axial[0] > 0

        loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
        initial_angle = 360 / (2 * np.pi) * np.arcsin(v_transverse[0] / v[0])

        if abs(initial_angle - loss_cone_angle) < 1000:
            points_counter += 1
            if in_loss_cone and positive_z_velocity:  # right loss cone
                particle_type = 'right'
                points_right_counter += 1
            elif in_loss_cone and not positive_z_velocity:  # left loss cone
                particle_type = 'left'
                points_left_counter += 1
            elif not in_loss_cone:  # trapped
                particle_type = 'trapped'
                points_trapped_counter += 1

            # check loss cone criterion as a function of time, for varying Bz(t)
            Rm_dynamic = field_dict['B0'] * field_dict['Rm'] / Bz
            out_of_loss_cone = (v_transverse / v) ** 2 >= 1 / Rm_dynamic
            in_loss_cone = (v_transverse / v) ** 2 < 1 / Rm_dynamic
            is_particle_escaping_left = in_loss_cone * (v_axial < 0)
            is_particle_escaping_right = in_loss_cone * (v_axial >= 0)

            if particle_type == 'right':

                counter_right_particles += out_of_loss_cone

                ind_first_got_trapped = np.where(out_of_loss_cone)[0]
                if len(ind_first_got_trapped) > 0:
                    inds_considered_trapped = range(ind_first_got_trapped[0], len(t))
                    counter_right_particles2[inds_considered_trapped] += 1.0

            elif particle_type == 'left':

                counter_left_particles += out_of_loss_cone

                ind_first_got_trapped = np.where(out_of_loss_cone)[0]
                if len(ind_first_got_trapped) > 0:
                    inds_considered_trapped = range(ind_first_got_trapped[0], len(t))
                    counter_left_particles2[inds_considered_trapped] += 1.0

            else:

                counter_trapped_particles += is_particle_escaping_right

                ind_first_kicked_to_right = np.where(is_particle_escaping_right)[0]
                if len(ind_first_kicked_to_right) > 0:
                    inds_considered_kicked_right = range(ind_first_kicked_to_right[0], len(t))
                    counter_trapped_particles2[inds_considered_kicked_right] += 1.0

                counter_trapped_particles3 += is_particle_escaping_left

                ind_first_kicked_to_left = np.where(is_particle_escaping_left)[0]
                if len(ind_first_kicked_to_left) > 0:
                    inds_considered_kicked_left = range(ind_first_kicked_to_left[0], len(t))
                    counter_trapped_particles4[inds_considered_kicked_left] += 1.0

            # check if particle bounced from the first mirror
            # z_max = 1
            # z_min = 0
            z_max = 1.1
            z_min = -0.1
            vz0 = v_axial[0]
            inds_search = np.where(t <= 2 * settings['l'] / abs(vz0))[0]
            if particle_type == 'right':
                inds_escaped_cell = np.where(z[inds_search] / settings['l'] > z_max)[0]
                if len(inds_escaped_cell) == 0:
                    counter_right_trapped_first_cell += 1
            if particle_type == 'left':
                inds_escaped_cell = np.where(z[inds_search] / settings['l'] < z_min)[0]
                if len(inds_escaped_cell) == 0:
                    counter_left_trapped_first_cell += 1
            # elif particle_type == 'trapped' and vz0 > 0:
            if particle_type == 'trapped':
                inds_escaped_cell = np.where(z[inds_search] / settings['l'] > z_max)[0]
                if len(inds_escaped_cell) > 0:
                    counter_trapped_escaped_right_first_cell += 1
            # elif particle_type == 'trapped' and vz0 < 0:
            if particle_type == 'trapped':
                inds_escaped_cell = np.where(z[inds_search] / settings['l'] < z_min)[0]
                if len(inds_escaped_cell) > 0:
                    counter_trapped_escaped_left_first_cell += 1

            # # check if particle bounced from the first mirror
            # vz0 = v_axial[0]
            # ind_cell = len(t)
            # inds_escape_right_cell = np.where(z / settings['l'] > 1.1)[0]
            # if len(inds_escape_right_cell) > 0:
            #     ind_cell = inds_escape_right_cell[0]
            # inds_escape_left_cell = np.where(z / settings['l'] < -0.1)[0]
            # if len(inds_escape_left_cell) > 0:
            #     ind_cell = min(ind_cell, inds_escape_left_cell[0])
            # inds_reverse_direction = np.where(v_axial / vz0 < 0)[0]
            # if len(inds_reverse_direction) > 0:
            #     ind_cell = min(ind_cell, inds_reverse_direction[0])
            # inds_search = range(ind_cell)
            # if particle_type == 'right':
            #     inds_escaped_cell = np.where(z[inds_search] / settings['l'] > 1)[0]
            #     if len(inds_escaped_cell) == 0:
            #         counter_right_trapped_first_cell += 1
            # if particle_type == 'left':
            #     inds_escaped_cell = np.where(z[inds_search] / settings['l'] < 0)[0]
            #     if len(inds_escaped_cell) == 0:
            #         counter_left_trapped_first_cell += 1
            # # elif particle_type == 'trapped' and vz0 > 0:
            # if particle_type == 'trapped':
            #     inds_escaped_cell = np.where(z[inds_search] / settings['l'] > 1)[0]
            #     if len(inds_escaped_cell) > 0:
            #         counter_trapped_escaped_right_first_cell += 1
            # # elif particle_type == 'trapped' and vz0 < 0:
            # if particle_type == 'trapped':
            #     inds_escaped_cell = np.where(z[inds_search] / settings['l'] < 0)[0]
            #     if len(inds_escaped_cell) > 0:
            #         counter_trapped_escaped_left_first_cell += 1

    #         # plot particle trajectories in (z,v_transverse) plane
    #         if particle_type == 'right':
    #             plt.figure(2)
    #             plt.plot(v_transverse / settings['v_th'], z / settings['l'], '-')
    #
    # # add the plot labels
    # plt.figure(2)
    # plt.ylabel('$z/l$')
    # plt.xlabel('$v_{\\perp}/v_{th}$')
    # plt.grid(True)
    # plt.tight_layout()

    print('#######')
    print('points_counter: ' + str(points_counter))
    print('right: ' + str(100.0 * points_right_counter / points_counter) + '%')
    print('left: ' + str(100.0 * points_left_counter / points_counter) + '%')
    print('trapped: ' + str(100.0 * points_trapped_counter / points_counter) + '%')

    # percent_right_particles = counter_right_particles / points_right_counter * 100.0
    # percent_right_particles2 = counter_right_particles2 / points_right_counter * 100.0
    # percent_left_particles = counter_left_particles / points_left_counter * 100.0
    # percent_left_particles2 = counter_left_particles2 / points_left_counter * 100.0
    # percent_trapped_particles = counter_trapped_particles / points_trapped_counter * 100.0
    # percent_trapped_particles2 = counter_trapped_particles2 / points_trapped_counter * 100.0
    # percent_trapped_particles3 = counter_trapped_particles3 / points_trapped_counter * 100.0
    # percent_trapped_particles4 = counter_trapped_particles4 / points_trapped_counter * 100.0

    percent_right_particles = counter_right_particles / points_counter * 100.0
    percent_right_particles2 = counter_right_particles2 / points_counter * 100.0
    percent_left_particles = counter_left_particles / points_counter * 100.0
    percent_left_particles2 = counter_left_particles2 / points_counter * 100.0
    percent_trapped_particles = counter_trapped_particles / points_counter * 100.0
    percent_trapped_particles2 = counter_trapped_particles2 / points_counter * 100.0
    percent_trapped_particles3 = counter_trapped_particles3 / points_counter * 100.0
    percent_trapped_particles4 = counter_trapped_particles4 / points_counter * 100.0

    print('#######')
    print('right particles getting trapped in first cell: ' + str(
        100.0 * counter_right_trapped_first_cell / points_counter) + '%')
    print('left particles getting trapped in first cell: ' + str(
        100.0 * counter_left_trapped_first_cell / points_counter) + '%')
    print('trapped particles escape to the right in first cell: ' + str(
        100.0 * counter_trapped_escaped_right_first_cell / points_counter) + '%')
    print('trapped particles escape to the left in first cell: ' + str(
        100.0 * counter_trapped_escaped_left_first_cell / points_counter) + '%')

    # t_axis = t / field_dict['tau_cyclotron']
    t_axis = t / (settings['l'] / settings['v_th'])
    # t_axis = t * 1e6

    plt.figure(1)

    # label_RF_params = ', for $v_{z,res}/v_{th}$=' + str(vz_res) + ', $\\alpha=\\omega_{RF}/\\omega_{cyc0}=$' + str(
    #     alpha) + ', $v_{RF}/v_{th}$=' + '{:.1f}'.format(v_RF)
    label = 'trapped or escaping left'
    # label = 'trapped/left' + label_RF_params
    # plt.plot(t_axis, percent_particles_trapped_or_left, '-', color=color, label=label)
    # color = 'r'
    plt.plot(t_axis, percent_right_particles, '-', color='b', label='born right, checked if trapped')
    plt.plot(t_axis, percent_right_particles2, '--', color='b',
             label='born right, once trapped considered trapped forever')
    plt.plot(t_axis, percent_trapped_particles, '-', color='r', label='born trapped, checked if kicked right')
    plt.plot(t_axis, percent_trapped_particles2, '--', color='r',
             label='born trapped, once kicked right considered right forever')
    plt.plot(t_axis, percent_trapped_particles3, '-', color='m', label='born trapped, checked if kicked left')
    plt.plot(t_axis, percent_trapped_particles4, '--', color='m',
             label='born trapped, once kicked left considered left forever')
    plt.plot(t_axis, percent_left_particles, '-', color='g', label='born left, checked if trapped')
    plt.plot(t_axis, percent_left_particles2, '--', color='g',
             label='born left, once trapped considered trapped forever')

    # t_single_cell = settings['l'] / settings['v_th'] * 1e6
    # plt.plot([t_single_cell, t_single_cell], [min(percent_particles_trapped_or_left_and_axis_bound), max(percent_particles_trapped_or_left_and_axis_bound)], '--', color='grey', label='t_single_cell')

    plt.xlabel('$t / (l / v_{th})$')
    # plt.xlabel('$t/\\tau_{cyc}$')
    # plt.xlabel('$t$ [$\\mu s$]')

    # plt.ylabel('% change of population')
    plt.ylabel('% change from total number of particles')
    # plt.title('$E_{RF}$=' + str(ERF) + 'kV/m' + ', Rm=' + str(Rm)
    #           + ', $v_{z,res}/v_{th}$=' + str(vz_res)
    #           + ', $\\alpha=\\omega_{RF}/\\omega_{cyc0}=$' + str(alpha)
    #           + ', $v_{RF}/v_{th}$=' + v_RF_label)
    plt.title('$v_{z,res}/v_{th}$=' + str(vz_res)
              + ', $\\alpha=\\omega_{RF}/\\omega_{cyc0}=$' + str(alpha_actual_label)
              + ', $v_{RF}/v_{th}$=' + v_RF_label)

    # plt.title('percent change of a population (right-going or trapped)')
    plt.grid(True)
    plt.tight_layout()
    # plt.legend()
