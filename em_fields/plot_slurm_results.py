

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.io import loadmat

plt.rcParams.update({'font.size': 12})

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set1/'

# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_0_detune_2'

# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_2'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_3_detune_2'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_5_detune_2'

# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_3'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_5_detune_3'

# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_1.5'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_3'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_4'

# set_name = 'tmax_1000_B0_0.3_Rm_3.0_T_3.0_traveling_ERF_2_detune_2'
# set_name = 'tmax_1000_B0_1.0_Rm_3.0_T_3.0_traveling_ERF_2_detune_2'

# save_dir += '/set2/'

# set_name = 'r0_0.1_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_3'
# set_name = 'r0_0.2_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_3'

# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_1_alpha_2_3'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_alpha_2_3'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_1_alpha_2_4'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_alpha_2_4'

# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_alpha_2'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_alpha_2.718'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_1_alpha_2_2.718'

# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_alpha_2.718'


# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_4_alpha_2'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_4_alpha_2.718'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_4_alpha_3.141'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_alpha_2_2.718'
# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_alpha_2.718_3.141'

# set_name = 'r0_0.0_z0_0.5_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_4_alpha_2.718'
# set_name = 'r0_0.0_z0_0.0_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_4_alpha_2.718'
# set_name = 'r0_0.0_z0_0.3_tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_4_alpha_2.718'

save_dir += '/set3/'
set_name = 'tmax_566_B0_0.1_T_3.0_traveling_ERF_0_alpha_2.718'

save_dir += set_name

# plt.close('all')

# v_abs_list = np.linspace(0.5, 1.5, 21)
# angle_to_z_axis_list = [i for i in range(0, 181, 5)]
# phase_RF_list = np.array([0, 0.25, 0.5]) * np.pi

v_abs_list = np.linspace(0.5, 1.5, 11)
angle_to_z_axis_list = [i for i in range(0, 181, 10)]
phase_RF_list = np.array([0]) * np.pi

# phase_RF = 0
# phase_RF = 0.25 * np.pi
# phase_RF = 0.5 * np.pi

# z_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list)])
# E_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list)])
# for ind_v_abs, v_abs in enumerate(v_abs_list):
#     for ind_angle, angle_to_z_axis in enumerate(angle_to_z_axis_list):
#         run_name = ''
#         run_name += 'v_' + '{:.2f}'.format(v_abs)
#         run_name += '_angle_' + str(int(angle_to_z_axis))  # later
#         run_name += '_phaseRF_' + '{:.2f}'.format(phase_RF / np.pi)
#
#         data = np.loadtxt(save_dir + '/' + run_name + '.txt')
#         z_mat[ind_v_abs, ind_angle] = data[0]
#         E_mat[ind_v_abs, ind_angle] = data[1]
# z_mat = z_mat.T
# E_mat = E_mat.T

compiled_mat_file = save_dir + '.mat'
compiled_mat_dict = loadmat(compiled_mat_file)
z_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list), len(phase_RF_list)])
E_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list), len(phase_RF_list)])
v_r_mean_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list), len(phase_RF_list)])
v_z_mean_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list), len(phase_RF_list)])
cnt = 0
for ind_v_abs, v_abs in enumerate(v_abs_list):
    for ind_angle, angle_to_z_axis in enumerate(angle_to_z_axis_list):
        for ind_phase, phase_RF in enumerate(phase_RF_list):
            z_mat[ind_v_abs, ind_angle, ind_phase] = compiled_mat_dict['z'][0][cnt]
            E_mat[ind_v_abs, ind_angle, ind_phase] = compiled_mat_dict['E'][0][cnt]
            v_r_mean_mat[ind_v_abs, ind_angle, ind_phase] = compiled_mat_dict['v_r_mean'][0][cnt]
            v_z_mean_mat[ind_v_abs, ind_angle, ind_phase] = compiled_mat_dict['v_z_mean'][0][cnt]
            cnt += 1

ind_phase = 0
# ind_phase = 2
z_mat = z_mat[:, :, ind_phase].T
E_mat = E_mat[:, :, ind_phase].T
v_r_mean_mat = v_r_mean_mat[:, :, ind_phase].T
v_z_mean_mat = v_z_mean_mat[:, :, ind_phase].T

# plot
# cmap = "YlGnBu"
cmap = None

# plt.subplot(1, 3, 1 + ind_phase)
# sns.heatmap(z_mat, xticklabels=v_abs_list, yticklabels=angle_to_z_axis_list)
v_abs_list_labels = ['{:.1f}'.format(v_abs) for v_abs in v_abs_list]
for i in range(len(v_abs_list_labels)):
    if np.mod(i, 2) == 1:
        v_abs_list_labels[i] = ''
angle_to_z_axis_list_labels = [str(int(angle_to_z_axis)) for i, angle_to_z_axis in enumerate(angle_to_z_axis_list)]
for i in range(len(angle_to_z_axis_list_labels)):
    if np.mod(i, 2) == 1:
        angle_to_z_axis_list_labels[i] = ''

# vmin = -30
# vmax = 30
vmin = None
vmax = None
# plt.figure(1)
# plt.figure()
# plt.figure(1, figsize=(13, 5))
# plt.figure(2, figsize=(13, 5))
plt.figure(figsize=(8, 3))
# plt.subplot(1, 2, 1)
plt.subplot(1, 4, 1)
# plt.subplot(3, 2, 5)
sns.heatmap(z_mat, xticklabels=v_abs_list_labels, yticklabels=angle_to_z_axis_list_labels, cmap=cmap, vmin=vmin,
            vmax=vmax)
plt.xlabel('$v/v_{th}$')
plt.ylabel('$\\theta_z$')
plt.title('$z$')
# plt.title(set_name)
# plt.tight_layout()

# vmin = 0.5
# vmax = 2
vmin = None
vmax = None
# plt.figure(2)
# plt.subplot(1, 2, 2)
plt.subplot(1, 4, 2)
# plt.subplot(3, 2, 6)
sns.heatmap(E_mat, xticklabels=v_abs_list_labels, cmap=cmap, vmin=vmin,
            vmax=vmax)
plt.xlabel('$v/v_{th}$')
# plt.ylabel('$\\theta_z$')
plt.title('$E$')

vmin = None
vmax = None
plt.subplot(1, 4, 3)
sns.heatmap(v_r_mean_mat, xticklabels=v_abs_list_labels, cmap=cmap, vmin=vmin,
            vmax=vmax)
# plt.xlabel('$v/v_{th}$')
plt.ylabel('$\\theta_z$')
plt.title('mean $v_r$')

vmin = None
vmax = None
plt.subplot(1, 4, 4)
sns.heatmap(v_z_mean_mat, xticklabels=v_abs_list_labels, cmap=cmap, vmin=vmin,
            vmax=vmax)
plt.xlabel('$v/v_{th}$')
# plt.ylabel('$\\theta_z$')
plt.title('mean $v_z$')

# plt.tight_layout()
