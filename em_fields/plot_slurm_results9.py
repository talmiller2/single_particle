import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 10})

# plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
save_dir += '/set24_B0_1T_l_3m_Post_Rm_3/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
E_RF_kVm = 10  # kV/m

RF_type = 'magnetic_transverse'
B_RF = 0.05  # T

use_RF = True
# use_RF = False

alpha_loop_list = np.round(np.linspace(0.7, 1.3, 15), 2)
lambda_RF_loop_list = np.round(np.linspace(-20, 20, 10), 0)

cnt_RF_params = 0
totol_loop_runs = len(lambda_RF_loop_list) * len(alpha_loop_list)

right_trapped_first_cell = np.nan * np.zeros([len(lambda_RF_loop_list), len(alpha_loop_list)])
left_trapped_first_cell = np.nan * np.zeros([len(lambda_RF_loop_list), len(alpha_loop_list)])
trapped_escaped_right_first_cell = np.nan * np.zeros([len(lambda_RF_loop_list), len(alpha_loop_list)])
trapped_escaped_left_first_cell = np.nan * np.zeros([len(lambda_RF_loop_list), len(alpha_loop_list)])

for ind_lambda, lambda_RF in enumerate(lambda_RF_loop_list):
    for ind_alpha, alpha_RF in enumerate(alpha_loop_list):

        try:
            cnt_RF_params += 1
            print('$$$$$ ' + str(cnt_RF_params) + ' / ' + str(totol_loop_runs) + ' $$$$$')

            print('lambda = ' + str(lambda_RF) + '[m], alpha = ' + str(alpha_RF))

            set_name = ''
            if use_RF is False:
                set_name += 'without_RF'
            else:
                if RF_type == 'electric_transverse':
                    set_name += 'ERF_' + str(E_RF_kVm)
                elif RF_type == 'magnetic_transverse':
                    set_name += 'BRF_' + str(B_RF)
                set_name += '_alpha_' + str(alpha_RF)
                set_name += '_lambda_' + str(lambda_RF)

            save_dir_curr = save_dir + set_name

            # load runs data
            data_dict_file = save_dir_curr + '.pickle'
            with open(data_dict_file, 'rb') as fid:
                data_dict = pickle.load(fid)
            settings = data_dict['settings']
            field_dict = data_dict['field_dict']

            # draw trajectories for several particles
            num_particles = len(data_dict['z'])
            # ind_points = range(1000)
            ind_points = range(num_particles)

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
                    # Rm_dynamic = field_dict['B0'] * field_dict['Rm'] / Bz
                    # out_of_loss_cone = (v_transverse / v) ** 2 >= 1 / Rm_dynamic
                    # in_loss_cone = (v_transverse / v) ** 2 < 1 / Rm_dynamic
                    # is_particle_escaping_left = in_loss_cone * (v_axial < 0)
                    # is_particle_escaping_right = in_loss_cone * (v_axial >= 0)

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
                    if particle_type == 'trapped' and vz0 > 0:
                        # if particle_type == 'trapped':
                        inds_escaped_cell = np.where(z[inds_search] / settings['l'] > z_max)[0]
                        if len(inds_escaped_cell) > 0:
                            counter_trapped_escaped_right_first_cell += 1
                    if particle_type == 'trapped' and vz0 < 0:
                        # if particle_type == 'trapped':
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

            right_trapped_first_cell[ind_lambda, ind_alpha] = 100.0 * counter_right_trapped_first_cell / points_counter
            left_trapped_first_cell[ind_lambda, ind_alpha] = 100.0 * counter_left_trapped_first_cell / points_counter
            trapped_escaped_right_first_cell[
                ind_lambda, ind_alpha] = 100.0 * counter_trapped_escaped_right_first_cell / points_counter
            trapped_escaped_left_first_cell[
                ind_lambda, ind_alpha] = 100.0 * counter_trapped_escaped_left_first_cell / points_counter

            # print('points_counter = ' + str(points_counter))
            # print('points_trapped_counter = ' + str(points_trapped_counter))
            # print('trapped_percent = ' + str(100.0 * points_trapped_counter / points_counter))

        except:
            print('FAILED set_name: ' + str(set_name))

# right_selectivity = right_trapped_first_cell / (trapped_escaped_right_first_cell + 1e-2)
# left_selectivity = left_trapped_first_cell / (trapped_escaped_left_first_cell + 1e-2)
right_selectivity = right_trapped_first_cell / (trapped_escaped_right_first_cell + 0)
left_selectivity = left_trapped_first_cell / (trapped_escaped_left_first_cell + 0)

# 2d plots

x_array = lambda_RF_loop_list
x_label = '$\\lambda$ [m]'
y_array = alpha_loop_list

# vmax_right = np.nanpercentile(right_selectivity, 95)
# vmax_left = np.nanpercentile(left_selectivity, 95)

# plt.figure(1)
# plt.figure(1, figsize=(15, 8))
plt.figure(1, figsize=(15, 7))
# plt.subplot(2, 3, 1)
plt.subplot(3, 3, 1)
sns.heatmap(right_trapped_first_cell.T, xticklabels=x_array, yticklabels=y_array)
# plt.xlabel(x_label)
# plt.ylabel('$\\alpha$')
# plt.title('right_trapped_first_cell %')
plt.title('% initially right got trapped')
# plt.tight_layout()
print('max right_trapped_first_cell = ' + str(np.nanmax(right_trapped_first_cell)))

# plt.figure(2)
# plt.subplot(2, 3, 2)
plt.subplot(3, 3, 2)
sns.heatmap(trapped_escaped_right_first_cell.T, xticklabels=x_array, yticklabels=y_array)
# plt.xlabel(x_label)
# plt.ylabel('$\\alpha$')
# plt.title('trapped_escaped_right_first_cell %')
plt.title('% initially trapped escaped right')
# plt.tight_layout()
print('max trapped_escaped_right_first_cell = ' + str(np.nanmax(trapped_escaped_right_first_cell)))

# plt.figure(3)
# plt.subplot(2, 3, 3)
plt.subplot(3, 3, 3)
sns.heatmap(right_selectivity.T, xticklabels=x_array, yticklabels=y_array)
# plt.xlabel(x_label)
# plt.ylabel('$\\alpha$')
plt.title('right-selectivity ratio')
# plt.tight_layout()
print('max right_selectivity = ' + str(np.nanmax(right_selectivity)))

# plt.subplot(2, 3, 4)
plt.subplot(3, 3, 4)
sns.heatmap(left_trapped_first_cell.T, xticklabels=x_array, yticklabels=y_array)
# plt.xlabel(x_label)
# plt.ylabel('$\\alpha$')
# plt.title('left_trapped_first_cell %')
plt.title('% initially left got trapped')
# plt.tight_layout()
print('max left_trapped_first_cell = ' + str(np.nanmax(left_trapped_first_cell)))

# plt.figure(2)
# plt.subplot(2, 3, 5)
plt.subplot(3, 3, 5)
sns.heatmap(trapped_escaped_left_first_cell.T, xticklabels=x_array, yticklabels=y_array)
# plt.xlabel(x_label)
# plt.ylabel('$\\alpha$')
# plt.title('trapped_escaped_left_first_cell %')
plt.title('% initially trapped escaped left')
# plt.tight_layout()
print('max trapped_escaped_left_first_cell = ' + str(np.nanmax(trapped_escaped_left_first_cell)))

# plt.subplot(2, 3, 6)
plt.subplot(3, 3, 6)
sns.heatmap(left_selectivity.T, xticklabels=x_array, yticklabels=y_array)
# plt.xlabel(x_label)
# plt.ylabel('$\\alpha$')
plt.title('left-selectivity ratio')
# plt.tight_layout()
print('max left_selectivity = ' + str(np.nanmax(left_selectivity)))

plt.tight_layout(pad=0.5)

# plt.figure(10)
plt.subplot(3, 3, 8)
rls = right_selectivity / left_selectivity
# rls = right_selectivity / (left_selectivity + 1e-2)
# vmax = np.nanpercentile(rls, 95)
vmax = np.nanmax(rls)
sns.heatmap(rls.T, xticklabels=x_array, yticklabels=y_array, vmax=vmax)
plt.xlabel(x_label)
plt.ylabel('$\\alpha$')
# plt.title('right / left selectivity')
plt.title('right-selectivity / left-selectivity')
plt.tight_layout(pad=0.5)
print('max rls = ' + str(np.nanmax(rls)))
