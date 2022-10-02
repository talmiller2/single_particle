import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

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
save_dir += '/set36_B0_1T_l_1m_Post_Rm_3_intervals/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 25  # kV/m
E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m
#
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
# alpha = 1.1
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
# beta = -3.7
# beta = -4.0
# beta = -4.5
# beta = -5.0
# beta = -6.0
# beta = -7.5
# beta = -8.0
# beta = -9.0
# beta = -10.0
# beta = 1.0
# beta = 3.0


# # set A
# alpha = 1.0
# beta = 0.0
# set_num = 'a'

# # set B
# alpha = 0.9
# beta = -1.0
# set_num = 'b'

# # set C
# alpha = 0.8
# beta = -5.0
# set_num = 'c'

# # set D
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

# number_of_time_intervals = 3
number_of_time_intervals = data_dict['t'].shape[1]

from matplotlib import cm

# colors = cm.rainbow(np.linspace(0, 1, number_of_time_intervals))
# # colors = ['b', 'g', 'r']
# # colors = ['r', 'g', 'b']
#
# for ind_t in range(number_of_time_intervals):
#     # for ind_t in [0, 10]:
#     # for ind_t in [0, 1]:
#     # for ind_t in [0, 10, 20]:
#     #     print(ind_t)
#
#     # inds_particles = range(data_dict['t'].shape[0])
#     # inds_particles = [0, 1, 2]
#     # inds_particles = range(1001)
#     inds_particles = range(100)
#
#     if ind_t == 0:
#         print('num particles = ' + str(len(inds_particles)))
#
#     v = data_dict['v'][inds_particles, ind_t]
#     v0 = data_dict['v'][inds_particles, 0]
#     vt = data_dict['v_transverse'][inds_particles, ind_t]
#     vt0 = data_dict['v_transverse'][inds_particles, 0]
#     vz = data_dict['v_axial'][inds_particles, ind_t]
#     vz0 = data_dict['v_axial'][inds_particles, 0]
#     theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
#     Bz = data_dict['Bz'][inds_particles, ind_t]
#     Bz0 = data_dict['Bz'][inds_particles, 0]
#     vt_adjusted = vt * np.sqrt(Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)
#
#     # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt0 ** 2.0 * (Bz / Bz0 - 1))
#     # vz_adjusted = np.sign(vz0) * np.sqrt(vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz))
#     # theta_adjusted = np.mod(360 / (2 * np.pi) * np.arctan(vt_adjusted / vz_adjusted), 180)
#
#     det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
#     inds_positive = np.where(det > 0)[0]
#     vz_adjusted = np.zeros(len(inds_particles))
#     vz_adjusted[inds_positive] = np.sign(vz0[inds_positive]) * np.sqrt(det[inds_positive])
#     # vz_adjusted[inds_positive] = np.sign(vz[inds_positive]) * np.sqrt(det[inds_positive])
#
#     theta_adjusted = 90.0 * np.ones(len(inds_particles))
#     theta_adjusted[inds_positive] = np.mod(
#         360 / (2 * np.pi) * np.arctan(vt_adjusted[inds_positive] / vz_adjusted[inds_positive]), 180)
#
#     color = colors[ind_t]
#
#     if ind_t == 0:
#         fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7), num=1)
#     if np.mod(ind_t, 5) == 1:
#         label = str(ind_t)
#         label = '$t \\cdot v_{th} / l$=' + '{:.2f}'.format(
#             data_dict['t'][0, ind_t] / (settings['l'] / settings['v_th']))
#         ax1.scatter(vz_adjusted / settings['v_th'], vt_adjusted / settings['v_th'], color=color, alpha=0.2, label=label)
#         if ind_t == 0:
#             # plot the diagonal LC lines
#             vz_axis = np.array([0, 2 * settings['v_th']])
#             vt_axis = vz_axis * np.sqrt(1 / (field_dict['Rm'] - 1))
#             ax1.plot(vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
#             ax1.plot(-vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
# ax1.set_xlabel('$v_z / v_{th}$')
# ax1.set_ylabel('$v_{\\perp} / v_{th}$')
# ax1.legend()
# ax1.grid(True)


# fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7), num=2)
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6), num=2)
num_particles = 1000
# colors = cm.rainbow(np.linspace(0, 1, num_particles))

dist_v_list = []
for ind_p in range(num_particles):
    v = data_dict['v'][ind_p, :]
    v0 = data_dict['v'][ind_p, 0]
    vt = data_dict['v_transverse'][ind_p, :]
    vt0 = data_dict['v_transverse'][ind_p, 0]
    vz = data_dict['v_axial'][ind_p, :]
    vz0 = data_dict['v_axial'][ind_p, 0]
    theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
    Bz = data_dict['Bz'][ind_p, :]
    Bz0 = data_dict['Bz'][ind_p, 0]
    vt_adjusted = vt * np.sqrt(Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

    det = vz ** 2.0 + vt ** 2.0 * (1 - Bz0 / Bz)
    inds_positive = np.where(det > 0)[0]
    vz_adjusted = np.zeros(len(vz))
    vz_adjusted[inds_positive] = np.sign(vz0) * np.sqrt(det[inds_positive])

    dist_v = max(np.sqrt((vz_adjusted - vz_adjusted[0]) ** 2 + (vt_adjusted - vt_adjusted[0]) ** 2))
    dist_v /= np.sqrt((vz_adjusted[0]) ** 2 + (vt_adjusted[0]) ** 2)
    dist_v_list += [dist_v]
max_dist_v = np.percentile(dist_v_list, 90)

for ind_p in range(num_particles):

    v = data_dict['v'][ind_p, :]
    v0 = data_dict['v'][ind_p, 0]
    vt = data_dict['v_transverse'][ind_p, :]
    vt0 = data_dict['v_transverse'][ind_p, 0]
    vz = data_dict['v_axial'][ind_p, :]
    vz0 = data_dict['v_axial'][ind_p, 0]
    theta = np.mod(360 / (2 * np.pi) * np.arctan(vt / vz), 180)
    Bz = data_dict['Bz'][ind_p, :]
    Bz0 = data_dict['Bz'][ind_p, 0]
    vt_adjusted = vt * np.sqrt(Bz0 / Bz)  # no need to adjust v to B_min because energy is conserved (assuming no RF)

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
    dist_v /= settings['v_th']
    dist_v *= 1.2
    # dist_v /= 2
    dist_v /= max_dist_v
    color = cm.rainbow(dist_v)

    ax2.plot(vz_adjusted / settings['v_th'], vt_adjusted / settings['v_th'],
             # color=colors[ind_p],
             color=color,
             alpha=0.2,
             )
    ax2.plot(vz_adjusted[0] / settings['v_th'], vt_adjusted[0] / settings['v_th'],
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
        vz_axis = np.array([0, 2 * settings['v_th']])
        vt_axis = vz_axis * np.sqrt(1 / (field_dict['Rm'] - 1))
        ax2.plot(vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')
        ax2.plot(-vz_axis / settings['v_th'], vt_axis / settings['v_th'], color='k', linestyle='--')

text = '(' + set_num + ')'
# text = '(b)'
ax2.text(0.13, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='right', verticalalignment='top',
         transform=fig2.axes[0].transAxes)

ax2.set_xlabel('$v_z / v_{th}$')
ax2.set_ylabel('$v_{\\perp} / v_{th}$')
# ax2.set_title(title)
ax2.set_xlim([-2.0, 2.0])
# ax2.set_ylim([0, 2.0])
ax2.set_ylim([0, 2.5])
fig2.set_tight_layout(True)
# ax2.legend()
ax2.grid(True)

## save plots to file
# save_dir = '../../../Papers/texts/paper2022/pics/'
# # file_name = 'v_space_evolution_' + set_name
# file_name = 'v_space_evolution_set_' + set_num
# if RF_type == 'magnetic_transverse':
#     file_name += '_BRF'
# beingsaved = plt.figure(2)
# # # beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
# beingsaved.savefig(save_dir + file_name + '.jpeg', format='jpeg', dpi=300)
