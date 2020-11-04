import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from em_fields.magnetic_forms import *

plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'axes.labelpad': 15})
# plt.rcParams.update({'lines.linestyle': '-'})
# # plt.rcParams.update({'lines.linestyle': '--'})

plt.close('all')

# save_dir = '../runs/set1/'
save_dir = '../runs/set2/'

os.makedirs(save_dir, exist_ok=True)

# RF definitions
E_RF_kV = 3  # kV/m
E_RF = E_RF_kV * 1e3  # the SI units is V/m

alpha_detune_list = [1.1, 1.5, 2.0]
# alpha_detune = alpha_detune_list[0]
# alpha_detune = alpha_detune_list[1]
alpha_detune = alpha_detune_list[2]

angle_to_z_axis_list = np.linspace(0, 180, 19)

# RF_type = 'uniform'
RF_type = 'traveling'

z0_list = np.linspace(0, 0.5, 6)
# z0 = z0_list[0]
# z0 = z0_list[-1]
# z0 = z0_list[1]
z0 = z0_list[3]

r0_list = [0, 0.1, 0.2]
r0 = r0_list[0]
# r0 = r0_list[1]
# r0 = r0_list[2]

colors = cm.rainbow(np.linspace(0, 1, len(angle_to_z_axis_list)))

for i, angle_to_z_axis in enumerate(angle_to_z_axis_list):

    # # run name
    # run_name = 'RF_' + str(RF_type)
    # run_name += '_' + '{:.0f}'.format(E_RF_kV) + '_kV'
    # if RF_type == 'traveling':
    #     run_name += '_alpha_' + str(alpha_detune)
    # run_name += '_angle_' + '{:.0f}'.format(angle_to_z_axis)
    # run_name += '_z0_' + '{:.2f}'.format(z0)
    # print('run_name:', run_name)

    # run name
    run_name = 'RF_' + str(RF_type)
    run_name += '_' + '{:.0f}'.format(E_RF_kV) + '_kV'
    if RF_type == 'traveling':
        run_name += '_alpha_detune_' + str(alpha_detune)
    run_name += '_angle_' + '{:.0f}'.format(angle_to_z_axis)
    run_name += '_z0_' + '{:.1f}'.format(z0)
    run_name += '_r0_' + '{:.1f}'.format(r0)
    print('run_name:', run_name)

    hist_file = save_dir + '/' + run_name + '.pickle'
    with open(hist_file, 'rb') as fid:
        hist = pickle.load(fid)

    t = hist['t'] / hist['tau_cyclotron']
    x = hist['x'][:, 0] / hist['l']
    y = hist['x'][:, 1] / hist['l']
    z = hist['x'][:, 2] / hist['l']
    vx = hist['v'][:, 0] / hist['v_th']
    vy = hist['v'][:, 1] / hist['v_th']
    vz = hist['v'][:, 2] / hist['v_th']
    # E = hist['E']
    # B = hist['B']

    R = np.sqrt(x ** 2 + y ** 2)
    v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    E_kin = 0.5 * v_norm ** 2.0

    # Plots
    linewidth = 2
    color = colors[i]
    label_prefix = '$\\theta=$' + str(angle_to_z_axis) + '$\degree$ '

    # plt.figure(1)
    # plt.plot(t, z, label=label_prefix+'$z/l$', linestyle='-', linewidth=linewidth, color=color)
    # plt.plot(t, E_kin / E_kin[0], label=label_prefix+'$E/E_0$', linestyle='--', linewidth=linewidth, color=color)
    # plt.xlabel('t / $\\tau_{cyc}$')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()

    plt.figure(1)
    # plt.plot(t, z, label=label_prefix+'$z/l$', linestyle='-', linewidth=linewidth, color=color)
    plt.plot(t, z, label=label_prefix, linestyle='-', linewidth=linewidth, color=color)
    plt.xlabel('t / $\\tau_{cyc}$')
    plt.ylabel('$z / l$')
    plt.grid(True)
    # plt.tight_layout()
    plt.legend()

    plt.figure(2)
    plt.plot(t, R, label=label_prefix, linestyle='-', linewidth=linewidth, color=color)
    plt.xlabel('t / $\\tau_{cyc}$')
    plt.ylabel('$R / l$')
    plt.grid(True)
    # plt.tight_layout()
    plt.legend()

    plt.figure(3)
    # plt.plot(t, E_kin / E_kin[0], label=label_prefix+'$E/E_0$', linestyle='-', linewidth=linewidth, color=color)
    plt.plot(t, E_kin / E_kin[0], label=label_prefix, linestyle='-', linewidth=linewidth, color=color)
    plt.xlabel('t / $\\tau_{cyc}$')
    plt.ylabel('$E / E_0$')
    plt.grid(True)
    # plt.tight_layout()
    # plt.constr
    plt.legend()
