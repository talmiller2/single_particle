import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 8})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

# figsize_large = (16, 9)
figsize_large = (14, 7)

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
save_dir += '/set34_B0_1T_l_3m_Post_Rm_3_intervals/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 25  # kV/m
# E_RF_kVm = 50  # kV/m
E_RF_kVm = 100  # kV/m

RF_type = 'magnetic_transverse'
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

alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set34
beta_loop_list = np.round(np.linspace(-5, 5, 11), 2)

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
alpha = 0.9
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
# beta = -4.0
# beta = -4.5
beta = -5.0
# beta = -6.0
# beta = -7.5
# beta = -8.0
# beta = -9.0
# beta = -10.0
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

number_of_time_intervals = data_dict['t'].shape[1]

counter_right_to_trapped = np.zeros(number_of_time_intervals)
counter_trapped_to_right = np.zeros(number_of_time_intervals)
counter_trapped_to_left = np.zeros(number_of_time_intervals)
counter_left_to_trapped = np.zeros(number_of_time_intervals)

from matplotlib import cm

colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
# colors = ['b', 'g', 'r']
# colors = ['r', 'g', 'b']

for ind_t in range(number_of_time_intervals):

    inds_particles = range(data_dict['t'].shape[0])
    if ind_t == 0:
        print('num particles = ' + str(len(inds_particles)))

        v0 = data_dict['v'][inds_particles, 0]
        vt0 = data_dict['v_transverse'][inds_particles, 0]
        vz0 = data_dict['v_axial'][inds_particles, 0]
        Bz0 = data_dict['Bz'][inds_particles, 0]

    v = data_dict['v'][inds_particles, ind_t]
    vt = data_dict['v_transverse'][inds_particles, ind_t]
    vz = data_dict['v_axial'][inds_particles, ind_t]
    Bz = data_dict['Bz'][inds_particles, ind_t]

    Bmax = field_dict['B0'] * field_dict['Rm']
    local_loss_cone_condition = (vt / v) ** 2.0 - Bz / Bmax

    # TODO: order particles by their initial population, and then track each particle as to where it went as a function of time

    if ind_t == 0:
        local_loss_cone_condition_ini = (vt0 / v0) ** 2.0 - 1 / field_dict['Rm']

        inds_ini = []
        num_particles_ini = {}
        num_particles_ini['trapped'] = 0
        num_particles_ini['right'] = 0
        num_particles_ini['left'] = 0
        for ind_p in inds_particles:
            if local_loss_cone_condition_ini[ind_p] >= 0:
                inds_ini += ['trapped']
                num_particles_ini['trapped'] += 1
            elif vz0[ind_p] >= 0:
                inds_ini += ['right']
                num_particles_ini['right'] += 1
            else:
                inds_ini += ['left']
                num_particles_ini['left'] += 1

    # check condition of particles at time t
    for ind_p in inds_particles:
        if local_loss_cone_condition[ind_p] >= 0:
            if inds_ini[ind_p] == 'right':
                counter_right_to_trapped[ind_t] += 1
            elif inds_ini[ind_p] == 'left':
                counter_left_to_trapped[ind_t] += 1
        elif vz[ind_p] >= 0:
            if inds_ini[ind_p] == 'trapped':
                counter_trapped_to_right[ind_t] += 1
        elif vz[ind_p] < 0:
            if inds_ini[ind_p] == 'trapped':
                counter_trapped_to_left[ind_t] += 1

# normalize counters
counter_right_to_trapped /= num_particles_ini['right']
counter_trapped_to_right /= num_particles_ini['trapped']
counter_trapped_to_left /= num_particles_ini['trapped']
counter_left_to_trapped /= num_particles_ini['left']

t_array = data_dict['t'][0]
t_array /= settings['l'] / settings['v_th']

inds_t_array = range(len(t_array))
fig, axs = plt.subplots(1, 3, figsize=figsize_large, num=1)
fig.suptitle(title)
linestyle = '-'

ax = axs.ravel()[0]
color = 'b'
ax.plot(t_array, counter_right_to_trapped, color=color, linestyle=linestyle)

ax = axs.ravel()[1]
color = 'b'
ax.plot(t_array, counter_trapped_to_right, color=color, linestyle=linestyle)
color = 'r'
ax.plot(t_array, counter_trapped_to_left, color=color, linestyle=linestyle)

ax = axs.ravel()[2]
color = 'r'
ax.plot(t_array, counter_left_to_trapped, color=color, linestyle=linestyle)

for ax in axs.ravel():
    ax.grid(True)
    ax.set_xlabel('$t \\cdot v_{th} / l$')
    ax.set_ylim([0, 1])

fig.set_tight_layout({'pad': 0.5, 'rect': (0, 0, 1, 0.95)})
