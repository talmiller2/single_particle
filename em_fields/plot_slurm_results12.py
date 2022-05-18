import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np
import copy

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 10})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'



plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set26_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set28_B0_1T_l_10m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set29_B0_1T_l_3m_Post_Rm_2_first_cell_center_crossing/'
# save_dir += '/set30_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
save_dir += '/set31_B0_1T_l_3m_Post_Rm_3_intervals/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
E_RF_kVm = 25  # kV/m
# E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

RF_type = 'magnetic_transverse'
B_RF = 0.05  # T
# B_RF = 0.1  # T

use_RF = True
# use_RF = False
#
absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'
r_0 = 0
# r_0 = 1.0
# r_0 = 3.0

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
# beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.6, 1.0, 21), 2)  # set28
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
# beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

alpha_loop_list = np.round(np.linspace(0.8, 1.0, 11), 2)  # set31
beta_loop_list = np.round(np.linspace(-10, 0, 11), 2)

# for ind_beta, beta_RF in enumerate(beta_loop_list):
#     for ind_alpha, alpha_RF in enumerate(alpha_loop_list):
# ind_alpha = 0
# ind_alpha = 1
# ind_alpha = 2
# ind_alpha = 4
# ind_alpha = 5
# ind_alpha = 7
# ind_beta = 1
# ind_beta = 2
# ind_beta = 5
# ind_beta = 4
# ind_beta = 3
# alpha = alpha_loop_list[ind_alpha]
# beta = beta_loop_list[ind_beta]
#
# alpha = 0.8
# alpha = 0.82
# alpha = 0.85
# alpha = 0.86
# alpha = 0.9
# alpha = 0.92
alpha = 0.94
# alpha = 0.95
# alpha = 0.97
# alpha = 0.98
# alpha = 0.99
# alpha = 1.0
# alpha = 1.01
# alpha = 1.02
# alpha = 1.04
# alpha = 1.05
# alpha = 1.1

# beta = 0.0
# beta = -0.1
# beta = -0.2
# beta = -0.3
# beta = -0.4
# beta = -0.5
# beta = -0.7
# beta = -1.0
# beta = -2.0
# beta = -2.5
# beta = -3.0
# beta = -3.75
# beta = -4.5
beta = -5.0
# beta = -7.5
# beta = -8.0
# beta = -9.0
# beta = -10.0
# beta = 0.5
# beta = 1.0

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
    set_name = 'r0_' + str(r_0) + '_' + set_name

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

# normalize_by_tfin = True
normalize_by_tfin = False

# fig, (axs) = plt.subplots(2, 2, figsize=(10,6))
# fig, (axs) = plt.subplots(3, 2, figsize=(10, 9))
# fig, (axs) = plt.subplots(3, 2, figsize=(12, 7))
# fig, (axs) = plt.subplots(3, 1, figsize=(16, 9))
fig1, ax1 = plt.subplots(1, 1)

# divide the phase space by the angle
# TODO: later make a better division where the LC doesnt cross in the middle of each bin, by definition

# N_theta = 3
# N_theta = 6
N_theta = 9
dtheta = 180.0 / N_theta
theta_bins_max_list = np.linspace(dtheta, 180.0, N_theta)
theta_bins_min_list = np.linspace(0, 180.0 - dtheta, N_theta)
# fig, axs = plt.subplots(2, 3, figsize=(16, 9))
fig, axs = plt.subplots(3, 3, figsize=(13, 7))
# fig, axs = plt.subplots(2, 3, figsize=(13, 7))
# fig, axs = plt.subplots(1, 3, figsize=(13, 7))


# number_of_time_intervals = 3
number_of_time_intervals = data_dict['t'].shape[1]

# densities_dict = {}
# for i in range(N_theta):
#     densities_dict[i] = []

particles_counter_mat_3d = np.zeros([N_theta, N_theta, number_of_time_intervals])

from matplotlib import cm

colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
# colors = ['b', 'g', 'r']
# colors = ['r', 'g', 'b']

for ind_t in range(number_of_time_intervals):
    # for ind_t in [0, 10]:
    # for ind_t in [0, 1]:
    #     print(ind_t)

    inds_particles = range(data_dict['t'].shape[0])
    # inds_particles = [0, 1, 2]
    # inds_particles = range(1001)

    v = data_dict['v'][inds_particles, ind_t]
    v0 = data_dict['v'][inds_particles, 0]
    vt = data_dict['v_transverse'][inds_particles, ind_t]
    vt0 = data_dict['v_transverse'][inds_particles, 0]
    vz = data_dict['v_axial'][inds_particles, ind_t]
    vz0 = data_dict['v_axial'][inds_particles, 0]
    theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
    theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
    Bz = data_dict['Bz'][inds_particles, ind_t]
    Bz0 = data_dict['Bz'][inds_particles, 0]
    vt_adjusted = vt * np.sqrt(Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

    # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt0 ** 2.0 * (Bz / Bz0 - 1))
    # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz))
    # theta_adjusted = np.mod(360 / (2 * np.pi) * np.arctan(vt_adjusted / vz_adjusted), 180)

    det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
    inds_positive = np.where(det > 0)[0]
    vz_adjusted = np.zeros(len(inds_particles))
    vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])
    # vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(det[inds_positive])

    theta_adjusted = 90.0 * np.ones(len(inds_particles))
    theta_adjusted[inds_positive] = np.mod(
        360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)

    # v_adjusted = np.sqrt(vt_adjusted ** 2.0 + vz_adjusted ** 2.0)

    # print('mean of vt_adjusted / vt0 = ' + str(np.mean(vt_adjusted) / np.mean(vt0)))
    # print('mean of vz_adjusted / vz0 = ' + str(np.mean(vz_adjusted) / np.mean(vz0)))
    # print('mean of v / vadj = ' + str(np.mean(v_adjusted) / np.mean(v)))
    # print('mean of v / v0 = ' + str(np.mean(v) / np.mean(v0)))

    # print('mean of vt_adjusted / vt0 = ' + str(np.mean(vt_adjusted / vt0)))
    # print('mean of vz_adjusted / vz0 = ' + str(np.mean(vz_adjusted / vz0)))
    # print('mean of v / vadj = ' + str(np.mean(v_adjusted / v)))
    # print('mean of v / v0 = ' + str(np.mean(v / v0)))

    color = colors[ind_t]
    # label = str(ind_t)
    label = '$t \\cdot v_{th} / l$=' + '{:.1f}'.format(data_dict['t'][0, ind_t] / (settings['l'] / settings['v_th']))
    ax1.scatter(vz_adjusted / settings['v_th'], vt_adjusted / settings['v_th'], color=color, alpha=0.2, label=label)
    if ind_t == 0:
        # plot the diagonal LC lines
        vz_axis = np.array([0, 2 * settings['v_th']])
        vt_axis = vz_axis * np.sqrt(1 / (field_dict['Rm'] - 1))
        ax1.plot(vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
        ax1.plot(-vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
    ax1.set_xlabel('$v_z / v_{th}$')
    ax1.set_ylabel('$v_{\\perp} / v_{th}$')
    ax1.legend()
    ax1.grid(True)

    # track where each particle originated
    if ind_t == 0:
        inds_bins_ini = []
        for ind_p in inds_particles:
            theta_curr = theta_adjusted[ind_p]
            ind_bin = [k for k, (t1, t2) in enumerate(zip(theta_bins_min_list, theta_bins_max_list))
                       if theta_curr > t1 and theta_curr <= t2][0]
            inds_bins_ini += [ind_bin]

    # track where each particle travelled in time t
    particles_counter_mat = np.zeros([N_theta, N_theta])
    for ind_p in inds_particles:
        theta_curr = theta_adjusted[ind_p]
        ind_bin_fin = [k for k, (t1, t2) in enumerate(zip(theta_bins_min_list, theta_bins_max_list))
                       if theta_curr > t1 and theta_curr <= t2][0]
        ind_bin_ini = inds_bins_ini[ind_p]
        particles_counter_mat[ind_bin_ini, ind_bin_fin] += 1

    if ind_t == 0:
        N0 = copy.deepcopy(np.diag(particles_counter_mat))
        # print(N0)

    # divide all densities by the parent initial density
    for i in range(N_theta):
        particles_counter_mat[i, :] /= (1.0 * N0[i])

    # print(particles_counter_mat)

    particles_counter_mat_3d[:, :, ind_t] = particles_counter_mat

    # for i in range(N_theta):
    #     densities_dict[i] += [len(inds_theta_bins[i])]

t_array = data_dict['t'][0]
t_array /= settings['l'] / settings['v_th']

# for i in range(N_theta):
#     densities_dict[i] = np.array(densities_dict[i])
#     densities_dict[i] = densities_dict[i] / (1.0 * densities_dict[i][0])


colors = cm.rainbow(np.linspace(0, 1, N_theta))
for i, ax in enumerate(axs.ravel()):
    for j in range(N_theta):
        ax.plot(t_array, particles_counter_mat_3d[i, j, :], color=colors[j])
    ax.set_title('bin #' + str(i + 1))
    # ax.set_xlabel('$t * v_{th} / l$')
fig.set_tight_layout(0.5)

## plot a heat map of all the rates
ind_t = 1
rates_mat = (particles_counter_mat_3d[:, :, ind_t] - particles_counter_mat_3d[:, :, 0]) / t_array[ind_t]
rates_mat = abs(rates_mat)
fig3, ax3 = plt.subplots(figsize=(7, 6))
sns.heatmap(rates_mat.T,
            # xticklabels=beta_loop_list, yticklabels=alpha_loop_list,
            # vmin=vmin, vmax=vmax,
            # annot=annot,
            # annot_kws={"fontsize": annot_fontsize}, fmt=annot_fmt,
            ax=ax3,
            )
ax.set_xlabel('bin ini')
ax.set_ylabel('bin fin')
