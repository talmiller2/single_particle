import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np
import copy

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

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
save_dir += '/set36_B0_1T_l_1m_Post_Rm_3_intervals/'

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


alpha_loop_list = np.round(np.linspace(0.8, 1.2, 21), 2)  # set36
beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)

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
# alpha = 0.855
# alpha = 0.86
# alpha = 0.9
# alpha = 0.92
# alpha = 0.94
# alpha = 0.95
# alpha = 0.96
# alpha = 0.98
# alpha = 0.99
# alpha = 1.0
# alpha = 1.01
# alpha = 1.02
# alpha = 1.04
# alpha = 1.05
# alpha = 1.08
# alpha = 1.1
# alpha = 1.12
alpha = 1.14
# alpha = 1.2

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
# beta = -4.0
# beta = -4.5
# beta = -5.0
# beta = -6.0
# beta = -7.5
# beta = -8.0
# beta = -9.0
# beta = -10.0
# beta = 1.0
beta = 1.5
# beta = 2.0
# beta = 4.0
# beta = 5.0

# # set A
# alpha = 1.0
# beta = 0.0

# # set B
# alpha = 0.8
# beta = -1.0

# # set C
# alpha = 0.9
# beta = -5.0

# # set D
# alpha = 0.9
# beta = -4.0

# # set E
# alpha = 0.9
# beta = -2.0

# set F
# alpha = 0.8
# beta = -5.0

# # set G
# alpha = 0.8
# beta = -3.5

alpha = 1.2
beta = 1.0
set_num = 'd'

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
    vt_adjusted = vt * np.sqrt(Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

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
t_array /= settings['l'] / settings['v_th']

colors = cm.rainbow(np.linspace(0, 1, N_theta))
nu_decay_list = []
nu_mat = np.zeros([N_theta, N_theta])

do_fit = True
# do_fit = False

inds_t_array = range(len(t_array))
fig, ax = plt.subplots(1, 1,
                       # figsize=(7, 7),
                       num=1)
fig.suptitle(title)

## calculate the saturation value to estimate the rate
# inds_t_saturation = range(7, 21)
# inds_t_saturation = range(2, 3)
inds_t_saturation = range(15, 31)
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

# ax.plot(t_array, particles_counter_mat2_3d[2, 1, :], color='g', linestyle='-', label='$\\bar{N}_{lc}$')
# ax.plot(t_array, particles_counter_mat2_3d[1, 0, :], color='r', linestyle='-', label='$\\bar{N}_{cr}$')
# ax.plot(t_array, particles_counter_mat2_3d[1, 2, :], color='orange', linestyle='-', label='$\\bar{N}_{cl}$')

ax.set_xlabel('$t \\cdot v_{th} / l$')
ax.set_ylim([0, 0.8])
ax.legend()
ax.grid(True)
# fig.set_tight_layout({'pad': 0.5, 'rect': (0, 0, 1, 0.95)})
