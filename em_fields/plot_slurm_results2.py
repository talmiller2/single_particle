import pickle

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

save_dir = '/Users/talmiller/Downloads/single_particle/'

save_dir += '/set4/'

# set_name = 'tmax_200_B0_0.1_T_3.0_traveling_ERF_0_alpha_2.718'
set_name = 'tmax_2003_B0_0.1_T_3.0_traveling_ERF_0_alpha_2.718_v2'
# set_name = 'tmax_200_B0_0.1_T_3.0_traveling_ERF_2_alpha_2.718'
# set_name = 'tmax_200_B0_0.1_T_3.0_traveling_ERF_4_alpha_2.718'

save_dir += set_name
# plt.close('all')

# load runs data

# mat_file = save_dir + '.mat'
# mat_dict = loadmat(mat_file)
# settings = mat_dict['settings']
# field_dict = mat_dict['field_dict']

data_dict_file = save_dir + '.pickle'
with open(data_dict_file, 'rb') as fid:
    data_dict = pickle.load(fid)
settings = data_dict['settings']
field_dict = data_dict['field_dict']

# draw trajectories for several particles
# ind_points = [0, 1, 2, 4, 5]
ind_points = [0]
# ind_points = [4]
# ind_points = range(5)
# ind_points = range(20, 30)
# ind_points = range(30, 40)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
fig = plt.figure(1, figsize=(16, 6))
if fig.axes == []:
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
else:
    [ax1, ax2, ax3] = fig.axes

for ind_point in ind_points:
    z = data_dict['z'][ind_point, :]
    v = data_dict['v'][ind_point, :]
    v_transverse = data_dict['v_transverse'][ind_point, :]
    v_axial = data_dict['v_axial'][ind_point, :]

    # calculate if a particle is initially in right loss cone
    # in_loss_cone = (E_transverse[0] / E[0] - 1 / Rm) > 0
    # positive_z_velocity = mat_dict['v_0'][ind_point, 2] > 0
    in_loss_cone = (v_transverse[0] / v[0] - field_dict['Rm'] ** (-0.5)) > 0
    positive_z_velocity = data_dict['v_0'][ind_point, 2] > 0

    if in_loss_cone and positive_z_velocity:
        linestyle = '-'
        linewidth = 2
    elif in_loss_cone and not positive_z_velocity:
        linestyle = ':'
        linewidth = 2
    else:
        linestyle = '--'
        linewidth = 1

    # plots
    ax1.plot(z / settings['l'], label=ind_point, linestyle=linestyle, linewidth=linewidth)
    ax1.set_xlabel('t')
    ax1.set_ylabel('$z/l$')
    ax1.legend()

    # ax2.plot(E / E_avg, label=ind_point, linestyle=linestyle, linewidth=linewidth)
    # ax2.set_xlabel('t')
    # ax2.set_ylabel('$E/E_{avg}$')
    # ax2.legend()
    #
    # ax3.plot(E_transverse / E_avg, label=ind_point, linestyle=linestyle, linewidth=linewidth)
    # ax3.set_xlabel('t')
    # ax3.set_ylabel('$E_{transverse}/E_{avg}$')
    # ax3.legend()

fig.tight_layout()
# fig.legend()
