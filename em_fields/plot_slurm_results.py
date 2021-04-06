

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

plt.rcParams.update({'font.size': 12})

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set1/'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_3_detune_2'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_2'
set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_3'
# set_name = 'tmax_1000_B0_0.1_Rm_3.0_T_3.0_traveling_ERF_2_detune_1.5'
save_dir += set_name

# plt.close('all')

v_abs_list = np.linspace(0.5, 1.5, 21)
angle_to_z_axis_list = [i for i in range(0, 181, 5)]
phase_RF_list = np.array([0, 0.25, 0.5]) * np.pi

# for ind_phase, phase_RF in enumerate(phase_RF_list):
phase_RF = 0
# phase_RF = 0.25 * np.pi
# phase_RF = 0.5 * np.pi

z_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list)])
E_mat = np.zeros([len(v_abs_list), len(angle_to_z_axis_list)])

for ind_v_abs, v_abs in enumerate(v_abs_list):
    for ind_angle, angle_to_z_axis in enumerate(angle_to_z_axis_list):
        run_name = ''
        run_name += 'v_' + '{:.2f}'.format(v_abs)
        run_name += '_angle_' + str(int(angle_to_z_axis))  # later
        run_name += '_phaseRF_' + '{:.2f}'.format(phase_RF / np.pi)

        data = np.loadtxt(save_dir + '/' + run_name + '.txt')
        z_mat[ind_v_abs, ind_angle] = data[0]
        E_mat[ind_v_abs, ind_angle] = data[1]
z_mat = z_mat.T
E_mat = E_mat.T

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

vmin = -30
vmax = 30
# plt.figure(1)
plt.figure(1, figsize=(13, 6))
# plt.figure(2, figsize=(13, 6))
plt.subplot(1, 2, 1)
sns.heatmap(z_mat, xticklabels=v_abs_list_labels, yticklabels=angle_to_z_axis_list_labels, cmap=cmap, vmin=vmin,
            vmax=vmax)
plt.xlabel('$v/v_{th}$')
plt.ylabel('$\\theta_z$')
plt.title('$z$')
# plt.title(set_name)
# plt.tight_layout()

vmin = 0.5
vmax = 2
# plt.figure(2)
plt.subplot(1, 2, 2)
sns.heatmap(E_mat, xticklabels=v_abs_list_labels, yticklabels=angle_to_z_axis_list_labels, cmap=cmap, vmin=vmin,
            vmax=vmax)
plt.xlabel('$v/v_{th}$')
plt.ylabel('$\\theta_z$')
plt.title('$E$')
# plt.tight_layout()
