import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns

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
# save_dir_main += '/set21_B0_1T_l_3m_Post_intervals_Rm_3_different_phases/'
save_dir_main += '/set22_B0_1T_l_3m_Post_intervals_Rm_3/'
# save_dir_main += '/set23_B0_1T_l_3m_Post_intervals_Rm_6/'

set_names = []

# Rm = 1.3
# Rm = 2
Rm = 3
# Rm = 4
# Rm = 6

# ERF = 0
# ERF = 1
# ERF = 2
# ERF = 5
# ERF = 10
# ERF = 20
# ERF = 25
# ERF = 30
ERF = 50
# ERF = 100

# v_loop_list = [0.5, 1.0, 1.5, 2.0]
# alpha_loop_list = [0.6, 1.0, 1.2, 1.5, 2.0]

v_loop_list = np.round(np.linspace(0.5, 2, 15), 2)
alpha_loop_list = np.round(np.linspace(0.6, 2, 15), 2)
alpha_loop_list_actual = alpha_loop_list * (1 + np.pi / 100)
v_RF = v_loop_list * alpha_loop_list_actual / (alpha_loop_list_actual - 1.0)

cnt_RF_params = 0
totol_loop_runs = len(v_loop_list) * len(alpha_loop_list)

right_trapped_first_cell = np.nan * np.zeros([len(v_loop_list), len(alpha_loop_list)])
left_trapped_first_cell = np.nan * np.zeros([len(v_loop_list), len(alpha_loop_list)])
trapped_escaped_right_first_cell = np.nan * np.zeros([len(v_loop_list), len(alpha_loop_list)])
trapped_escaped_left_first_cell = np.nan * np.zeros([len(v_loop_list), len(alpha_loop_list)])
lambda_RF_over_l = np.nan * np.zeros([len(v_loop_list), len(alpha_loop_list)])

for ind_v, vz_res in enumerate(v_loop_list):
    for ind_alpha, alpha in enumerate(alpha_loop_list):

        try:
            cnt_RF_params += 1
            print('$$$$$ ' + str(cnt_RF_params) + ' / ' + str(totol_loop_runs) + ' $$$$$')

            alpha_actual = alpha * (1 + np.pi / 100)
            v_RF = vz_res * alpha_actual / (alpha_actual - 1.0)
            v_RF_label = '{:.2f}'.format(v_RF)
            alpha_actual_label = '{:.2f}'.format(alpha_actual)

            print('vz_res/v_th = ' + str(vz_res) + ', alpha = ' + str(alpha))
            omega_RF_over_omega_cyc_0 = alpha
            print('omega_RF/omega_cyc0 = ' + '{:.2f}'.format(omega_RF_over_omega_cyc_0) + ', v_RF/v_th = ' + v_RF_label)


            if ERF > 0:
                set_name = 'Rm_' + str(int(Rm)) + '_ERF_' + str(ERF) + '_alpha_' + str(alpha) + '_vz_' + str(vz_res)
            else:
                set_name = 'Rm_' + str(int(Rm)) + '_ERF_0'

            save_dir = save_dir_main + set_name

            # load runs data
            data_dict_file = save_dir + '.pickle'
            with open(data_dict_file, 'rb') as fid:
                data_dict = pickle.load(fid)
            settings = data_dict['settings']
            field_dict = data_dict['field_dict']

            # RF wavelength
            vzres = field_dict['v_z_factor_list'][0] * settings['v_th']
            lambda_RF = 2 * np.pi / field_dict['omega_cyclotron'] * vzres / (alpha_actual - 1)
            RF_wave_direction = np.sign(lambda_RF)
            # print('lambda_RF / l = ' + '{:.2f}'.format(abs(lambda_RF / settings['l'])) + ' (direction ' + str(RF_wave_direction) + ')')
            lambda_RF_over_l[ind_v, ind_alpha] = lambda_RF / settings['l']

            # draw trajectories for several particles
            num_particles = len(data_dict['z'])
            ind_points = range(1000)

            points_counter = 0
            points_trapped_counter = 0
            counter_right_trapped_first_cell = 0
            counter_left_trapped_first_cell = 0
            counter_trapped_escaped_right_first_cell = 0
            counter_trapped_escaped_left_first_cell = 0

            for ind_point in ind_points:

                inds_trajectory = range(len(data_dict['t'][ind_point]))

                t = np.array(data_dict['t'][ind_point])[inds_trajectory]
                z = np.array(data_dict['z'][ind_point])[inds_trajectory]
                v = np.array(data_dict['v'][ind_point])[inds_trajectory]
                v_transverse = np.array(data_dict['v_transverse'][ind_point])[inds_trajectory]
                v_axial = np.array(data_dict['v_axial'][ind_point])[inds_trajectory]
                Bz = np.array(data_dict['Bz'][ind_point])[inds_trajectory]

                # calculate if a particle is initially in right loss cone
                in_loss_cone = (v_transverse[0] / v[0]) ** 2 < 1 / field_dict['Rm']
                positive_z_velocity = v_axial[0] > 0

                loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
                initial_angle = 360 / (2 * np.pi) * np.arcsin(v_transverse[0] / v[0])

                if abs(initial_angle - loss_cone_angle) < 1000:
                    points_counter += 1
                    if in_loss_cone and positive_z_velocity:  # right loss cone
                        particle_type = 'right'
                    elif in_loss_cone and not positive_z_velocity:  # left loss cone
                        particle_type = 'left'
                    elif not in_loss_cone:  # trapped
                        particle_type = 'trapped'
                        points_trapped_counter += 1

                    # check loss cone criterion as a function of time, for varying Bz(t)
                    Rm_dynamic = field_dict['B0'] * field_dict['Rm'] / Bz
                    out_of_loss_cone = (v_transverse / v) ** 2 >= 1 / Rm_dynamic
                    in_loss_cone = (v_transverse / v) ** 2 < 1 / Rm_dynamic
                    is_particle_escaping_left = in_loss_cone * (v_axial < 0)
                    is_particle_escaping_right = in_loss_cone * (v_axial >= 0)

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

            # print('#######')
            # print('right particles getting trapped in first cell: ' + str(
            #     100.0 * counter_right_trapped_first_cell / points_counter) + '%')
            # print('left particles getting trapped in first cell: ' + str(
            #     100.0 * counter_left_trapped_first_cell / points_counter) + '%')
            # print('trapped particles escape to the right in first cell: ' + str(
            #     100.0 * counter_trapped_escaped_right_first_cell / points_counter) + '%')
            # print('trapped particles escape to the left in first cell: ' + str(
            #     100.0 * counter_trapped_escaped_left_first_cell / points_counter) + '%')

            right_trapped_first_cell[ind_v, ind_alpha] = 100.0 * counter_right_trapped_first_cell / points_counter
            left_trapped_first_cell[ind_v, ind_alpha] = 100.0 * counter_left_trapped_first_cell / points_counter
            trapped_escaped_right_first_cell[
                ind_v, ind_alpha] = 100.0 * counter_trapped_escaped_right_first_cell / points_counter
            trapped_escaped_left_first_cell[
                ind_v, ind_alpha] = 100.0 * counter_trapped_escaped_left_first_cell / points_counter

            # print('points_counter = ' + str(points_counter))
            # print('points_trapped_counter = ' + str(points_trapped_counter))
            # print('trapped_percent = ' + str(100.0 * points_trapped_counter / points_counter))

        except:
            print('FAILED.')

right_selectivity = right_trapped_first_cell / (trapped_escaped_right_first_cell + 1e-2)
left_selectivity = left_trapped_first_cell / (trapped_escaped_left_first_cell + 1e-2)

# 2d plots

x_array = v_loop_list
x_label = '$v_{z,res}/v_{th}$'
# x_array = np.round(v_RF, 2)
# x_label = '$v_{RF}/v_{th}$'
# y_array = alpha_loop_list
y_array = np.round(alpha_loop_list_actual, 2)

vmax_right = np.nanpercentile(right_selectivity, 95)
vmax_left = np.nanpercentile(left_selectivity, 95)

# plt.figure(1)
plt.figure(1, figsize=(15, 8))
plt.subplot(2, 3, 1)
sns.heatmap(right_trapped_first_cell.T, xticklabels=x_array, yticklabels=y_array)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('right_trapped_first_cell %')
# plt.tight_layout()

# plt.figure(2)
plt.subplot(2, 3, 2)
sns.heatmap(trapped_escaped_right_first_cell.T, xticklabels=x_array, yticklabels=y_array)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('trapped_escaped_right_first_cell %')
# plt.tight_layout()

# plt.figure(3)
plt.subplot(2, 3, 3)
sns.heatmap(right_selectivity.T, xticklabels=x_array, yticklabels=y_array, vmax=vmax_right)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('right_selectivity')
# plt.tight_layout()

plt.subplot(2, 3, 4)
sns.heatmap(left_trapped_first_cell.T, xticklabels=x_array, yticklabels=y_array)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('left_trapped_first_cell %')
# plt.tight_layout()

# plt.figure(2)
plt.subplot(2, 3, 5)
sns.heatmap(trapped_escaped_left_first_cell.T, xticklabels=x_array, yticklabels=y_array)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('trapped_escaped_left_first_cell %')
# plt.tight_layout()

plt.subplot(2, 3, 6)
sns.heatmap(left_selectivity.T, xticklabels=x_array, yticklabels=y_array, vmax=vmax_left)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('left_selectivity')
# plt.tight_layout()

plt.tight_layout(pad=0.5)

plt.figure(10)
# rls = right_selectivity / left_selectivity
rls = right_selectivity / (left_selectivity + 1e-2)
vmax = np.nanpercentile(rls, 95)
# vmax = np.nanmax(rls)
sns.heatmap(rls.T, xticklabels=x_array, yticklabels=y_array, vmax=vmax)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('right / left selectivity')
plt.tight_layout(pad=0.5)

plt.figure(11)
sns.heatmap(lambda_RF_over_l.T, xticklabels=x_array, yticklabels=y_array)
# sns.heatmap(lambda_RF_over_l.T, xticklabels=x_array, yticklabels=y_array, vmin=0)
# sns.heatmap(lambda_RF_over_l.T, xticklabels=x_array, yticklabels=y_array, vmax=0)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
plt.title('$\\lambda_{RF}/l$')
plt.tight_layout(pad=0.5)
