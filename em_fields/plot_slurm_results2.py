from em_fields.default_settings import define_default_settings
from em_fields.em_functions import get_thermal_velocity, get_cyclotron_angular_frequency
from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

import numpy as np
from scipy.io import loadmat

plt.rcParams.update({'font.size': 12})

save_dir = '/Users/talmiller/Downloads/single_particle/'

save_dir += '/set4/'

# set_name = 'tmax_200_B0_0.1_T_3.0_traveling_ERF_0_alpha_2.718'
set_name = 'tmax_200_B0_0.1_T_3.0_traveling_ERF_2_alpha_2.718'
# set_name = 'tmax_200_B0_0.1_T_3.0_traveling_ERF_4_alpha_2.718'

save_dir += set_name
# plt.close('all')


settings = define_default_settings()
c = settings['c']
m = settings['mi']
q = settings['q']
T_keV = 3.0
T_eV = T_keV * 1e3
kB_eV = settings['kB_eV']
v_th = get_thermal_velocity(T_eV, m, kB_eV)
E_avg = 0.5 * settings['mi'] * v_th ** 2.0
l = 10.0  # m (interaction length)
B0 = 0.1  # Tesla
omega_cyclotron = get_cyclotron_angular_frequency(q, B0, m)
tau_cyclotron = 2 * np.pi / omega_cyclotron
Rm = 3.0

mat_file = save_dir + '.mat'
mat_dict = loadmat(mat_file)

# draw trajectories for several particles
# ind_points = [0, 1, 2, 4, 5]
# ind_points = [0]
# ind_points = [4]
ind_points = range(5)
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
    z = mat_dict['z'][ind_point, :]
    E = mat_dict['E'][ind_point, :]
    E_transverse = mat_dict['E_transverse'][ind_point, :]

    # calculate if a particle is initially in right loss cone
    in_loss_cone = (E_transverse[0] / E[0] - 1 / Rm) > 0
    positive_z_velocity = mat_dict['v_0'][ind_point, 2] > 0

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
    ax1.plot(z / l, label=ind_point, linestyle=linestyle, linewidth=linewidth)
    ax1.set_xlabel('t')
    ax1.set_ylabel('$z/l$')
    ax1.legend()

    ax2.plot(E / E_avg, label=ind_point, linestyle=linestyle, linewidth=linewidth)
    ax2.set_xlabel('t')
    ax2.set_ylabel('$E/E_{avg}$')
    ax2.legend()

    ax3.plot(E_transverse / E_avg, label=ind_point, linestyle=linestyle, linewidth=linewidth)
    ax3.set_xlabel('t')
    ax3.set_ylabel('$E_{transverse}/E_{avg}$')
    ax3.legend()

fig.tight_layout()
# fig.legend()
