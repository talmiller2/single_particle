import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import maxwell

from em_fields.RF_field_forms import E_RF_function, B_RF_function
from em_fields.default_settings import define_default_settings, define_default_field
from em_fields.em_functions import evolve_particle_in_em_fields

plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})

plt.close('all')
plot3d_exists = False
# plot3d_exists = True

for ind_sim in range(1):
    # for ind_sim in range(2):
    # for ind_sim in range(3):
    # for ind_sim in range(5):
    # for ind_sim in range(10):
    # for ind_sim in range(20):

    settings = {}
    settings['trajectory_save_method'] = 'intervals'
    settings['num_snapshots'] = 121
    # settings['num_snapshots'] = 10000

    settings = define_default_settings(settings)

    field_dict = {}
    field_dict['mirror_field_type'] = 'const'
    # field_dict['mirror_field_type'] = 'logan'

    # field_dict['RF_type'] = 'electric_transverse'
    # field_dict['E_RF_kVm'] = 0
    # field_dict['E_RF_kVm'] = 1
    field_dict['E_RF_kVm'] = 10
    # field_dict['E_RF_kVm'] = 20
    field_dict['RF_type'] = 'magnetic_transverse'
    # field_dict['B_RF'] = 0
    # field_dict['B_RF'] = 0.001
    # field_dict['B_RF'] = 0.01
    # field_dict['B_RF'] = 0.03
    field_dict['B_RF'] = 0.04
    # field_dict['B_RF'] = 0.06
    # field_dict['B_RF'] = 0.05
    # field_dict['B_RF'] = 0.2
    # field_dict['B_RF'] = 0.5
    # field_dict['B_RF'] = 0.02
    # field_dict['B_RF'] = 0.1
    # field_dict['alpha_RF_list'] = [1.0]
    # field_dict['alpha_RF_list'] = [1.1]
    # field_dict['alpha_RF_list'] = [1.5]
    # field_dict['alpha_RF_list'] = [0.9]
    # field_dict['alpha_RF_list'] = [0.8]
    # field_dict['alpha_RF_list'] = [0.5]
    # field_dict['alpha_RF_list'] = [0.88]
    # field_dict['alpha_RF_list'] = [0.92]
    # field_dict['alpha_RF_list'] = [0.94]
    # field_dict['alpha_RF_list'] = [0.95]
    # field_dict['alpha_RF_list'] = [0.96]
    # field_dict['alpha_RF_list'] = [0.98]
    # field_dict['alpha_RF_list'] = [0.99]
    field_dict['alpha_RF_list'] = [0.995]
    # field_dict['alpha_RF_list'] = [0.999]
    # field_dict['alpha_RF_list'] = [1.01]
    # field_dict['alpha_RF_list'] = [1.02]
    # field_dict['alpha_RF_list'] = [1.04]
    # field_dict['alpha_RF_list'] = [1.05]
    # field_dict['alpha_RF_list'] = [1.1]
    # field_dict['alpha_RF_list'] = [0.6]
    field_dict['beta_RF_list'] = [0]
    # field_dict['beta_RF_list'] = [-1.0]
    # field_dict['beta_RF_list'] = [-10.0]
    # field_dict['beta_RF_list'] = [-2.0]
    # field_dict['beta_RF_list'] = [1.0]
    # field_dict['beta_RF_list'] = [(field_dict['alpha_RF_list'][0] - 1) * 95043192 / (2 * np.pi * 982336)]

    # field_dict['anticlockwise'] = -1

    # field_dict['phase_RF_addition'] = 0
    # field_dict['phase_RF_addition'] = np.pi / 3
    field_dict['phase_RF_addition'] = 2 * np.pi * np.random.randn()

    field_dict['use_RF_correction'] = False

    field_dict = define_default_field(settings, field_dict)

    loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
    if ind_sim == 0:
        # angle = 0.99 * loss_cone_angle
        # angle = 0.5 * loss_cone_angle
        # angle = 0.2 * loss_cone_angle
        angle = 1.0
        # angle = 1.5 * loss_cone_angle
        # elif ind_sim == 1:
        #     # angle = 1.01 * loss_cone_angle
    else:
        field_dict['phase_RF_addition'] = 2 * np.pi * np.random.rand(ind_sim)[0]
        field_dict = define_default_field(settings, field_dict)
        print('phase_RF_addition = ' + str(field_dict['phase_RF_addition']))
        # angle = 90 * np.random.rand()
        # print('angle = ' + str(angle))
    unit_vec = np.random.randn(3)
    unit_vec /= np.linalg.norm(unit_vec)

    # sampling velocity from Maxwell-Boltzmann
    scale = np.sqrt(settings['kB_eV'] * settings['T_eV'] / settings['mi'])
    v_abs_sampled = maxwell.rvs(size=1, scale=scale)

    x = 0
    y = np.sin(angle / 360 * 2 * np.pi)
    z = np.cos(angle / 360 * 2 * np.pi)
    unit_vec = np.array([x, y, z]).T
    v_0 = settings['v_th'] * unit_vec
    # v_0 = v_abs_sampled * unit_vec
    # v_0[1] *= -1

    v_perp = np.sqrt(v_0[0] ** 2 + v_0[1] ** 2)
    cyclotron_radius = v_perp / field_dict['omega_cyclotron']

    x_0 = np.array([0, 2 * cyclotron_radius, 0])
    # if ind_sim == 0:
    #     x_0 = np.array([0, 2 * cyclotron_radius, 0])
    # elif ind_sim == 1:
    #     x_0 = np.array([1 * cyclotron_radius, 0, 0])
    #     v_0[1] *= -1
    # elif ind_sim == 2:
    #     x_0 = np.array([1 * cyclotron_radius, 0, -50 * cyclotron_radius])
    #     v_0[1] *= -1
    #     v_0[2] *= -1

    # settings['time_step_tau_cyclotron_divisions'] = 20
    settings['time_step_tau_cyclotron_divisions'] = 100
    # settings['time_step_tau_cyclotron_divisions'] = 300
    dt = field_dict['tau_cyclotron'] / settings['time_step_tau_cyclotron_divisions']
    # sim_cyclotron_periods = 5
    # sim_cyclotron_periods = 10
    # sim_cyclotron_periods = 20
    # sim_cyclotron_periods = 30
    # sim_cyclotron_periods = 50
    # sim_cyclotron_periods = 70
    # sim_cyclotron_periods = 100
    # sim_cyclotron_periods = 200
    sim_cyclotron_periods = 500
    t_max = sim_cyclotron_periods * field_dict['tau_cyclotron']
    num_steps = int(t_max / dt)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_RF_function, B_RF_function,
                                        num_steps=num_steps, q=settings['q'], m=settings['mi'], field_dict=field_dict)
    t = hist['t']
    t /= field_dict['tau_cyclotron']
    x = hist['x'][:, 0] / cyclotron_radius
    y = hist['x'][:, 1] / cyclotron_radius
    # x = hist['x'][:, 0] / settings['l']
    # y = hist['x'][:, 1] / settings['l']
    # z = hist['x'][:, 2] / settings['l']
    # z = hist['x'][:, 2] / cyclotron_radius
    z = hist['x'][:, 2]
    vx = hist['v'][:, 0]
    vy = hist['v'][:, 1]
    vz = hist['v'][:, 2]
    vx /= settings['v_th']
    vy /= settings['v_th']
    vz /= settings['v_th']
    v_abs = np.sqrt(hist['v'][:, 0] ** 2 + hist['v'][:, 1] ** 2 + hist['v'][:, 2] ** 2)
    energy_change = 100.0 * (v_abs ** 2.0 - v_abs[0] ** 2.0) / v_abs[0] ** 2.0

    # R = np.sqrt(x ** 2 + y ** 2)
    v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    vt = np.sqrt(vx ** 2 + vy ** 2)

    ### Plots
    if field_dict['RF_type'] == 'electric_transverse':
        label = '$E_{RF}$=' + str(field_dict['E_RF_kVm']) + 'kV/m'
    else:
        label = '$B_{RF}$=' + str(field_dict['B_RF']) + 'T'
    # label += ', $\\alpha=$' + str(field_dict['alpha_RF_list'][0])
    label += ', $\\omega_{RF} / \\omega_{cyc}=$' + str(field_dict['alpha_RF_list'][0])

    linewidth = 2

    label2 = '$\\theta_0$=' + '{:.1f}'.format(angle)
    plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.plot(t, x, label='x', linewidth=linewidth, color='b')
    # plt.plot(t, y, label='y', linewidth=linewidth, color='g')
    # plt.plot(t, z, label='z', linewidth=linewidth, color='r')
    # plt.plot(t, z, label='z', linewidth=linewidth)
    plt.plot(t, z, label=label2, linewidth=linewidth)
    # plt.plot(t, R, label='R', linewidth=linewidth, color='k')
    # plt.legend()
    # plt.xlabel('t')
    plt.xlabel('t/$\\tau_{cyc}$')
    # plt.ylabel('coordinate')
    # plt.ylabel('z/$r_{cyc}$')
    plt.ylabel('z [m]')
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()

    ### analytic model
    tr = t * field_dict['tau_cyclotron']
    omega_cyc = settings['e'] * field_dict['B0'] / settings['mi']
    omega_B = settings['e'] * field_dict['B_RF'] / settings['mi']
    # A = omega_B * vz[0] / (4 * omega_cyc + 2 * omega_B)
    # A *= 10
    # A = 0.466
    # A = 0.5
    A = 0.5 * vz[0]
    # A = v_norm[0] - vy[0]
    vx_model = -vy[0] * np.sin(omega_cyc * tr) \
               - A * np.sin((omega_cyc + omega_B) * tr) \
               + A * np.sin((omega_cyc - omega_B) * tr)
    vy_model = vy[0] * np.cos(omega_cyc * tr) \
               + A * np.cos((omega_cyc + omega_B) * tr) \
               - A * np.cos((omega_cyc - omega_B) * tr)
    vz_model = vz[0] * np.cos(omega_B * tr)

    plt.figure(4)
    # plt.subplot(1,2,2)
    plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
    plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
    # plt.plot(t, vz, label='$v_z$', linewidth=linewidth, color='r')
    plt.plot(t, vz, label='$v_z$ ($\\bar{v}_z/v_{z,0}=$' + '{:.3f}'.format(np.mean(vz) / vz[0]) + ')',
             linewidth=linewidth, color='r')
    # plt.plot(t, vx_model, label='$v_x$ model', linewidth=linewidth, color='b', linestyle='--')
    # plt.plot(t, vy_model, label='$v_y$ model', linewidth=linewidth, color='g', linestyle='--')
    # plt.plot(t, vz_model, label='$v_z$ model', linewidth=linewidth, color='r', linestyle='--')
    # plt.plot(t, vz, label='$v_z$', linewidth=linewidth)
    # plt.plot(t, vz, label=label2, linewidth=linewidth)
    # plt.plot(t, v_norm, label='$v_{norm}$', linewidth=linewidth, color='k')
    plt.legend()
    # plt.xlabel('t')
    plt.xlabel('t/$\\tau_{cyc}$')
    # plt.ylabel('v')
    plt.ylabel('v/$v_{th}$')
    # plt.ylabel('$v_z/v_{th}$')
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()

    ### Plot fourier transform
    plt.figure(5)
    inds_t = range(len(t))
    # inds_t = range(int(0 * len(t)), int(0.5 * len(t)))
    # inds_t = range(int(0 * len(t)), int(0.2 * len(t)))
    tf = t[inds_t]
    freq = [100.0 * i / len(tf) for i in list(range(len(tf)))]
    # plt.plot(freq, abs(np.fft.fft(vx[inds_t])), label='$v_x$', linewidth=linewidth, color='b')
    plt.plot(freq, abs(np.fft.fft(vy[inds_t])), label='$v_y$', linewidth=linewidth, color='g')
    plt.plot(freq, abs(np.fft.fft(vz[inds_t])), label='$v_z$', linewidth=linewidth, color='r')
    # inds_t = range(int(0.5 * len(t)), int(1.0 * len(t)))
    # # inds_t = range(int(0.2 * len(t)), int(0.4 * len(t)))
    # tf = t[inds_t]
    # freq = [100.0 * i / len(tf) for i in list(range(len(tf)))]
    # # plt.plot(freq, abs(np.fft.fft(vx[inds_t])), label='$v_x$', linewidth=linewidth, color='b')
    # plt.plot(freq, abs(np.fft.fft(vy[inds_t])), label='$v_y$', linewidth=linewidth, color='g', linestyle='--')
    # plt.plot(freq, abs(np.fft.fft(vz[inds_t])), label='$v_z$', linewidth=linewidth, color='r', linestyle='--')
    plt.legend()
    plt.xlabel('$\\omega/\\omega_{cyc}$')
    plt.xlim([0, 1.5])
    plt.title('Fourier transforms: ' + label)
    plt.grid(True)
    plt.tight_layout()

    #
    # plt.figure(2)
    # plt.plot(x, y, linewidth=linewidth)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.grid(True)
    # plt.tight_layout()
    #
    # plt.figure(3)
    # plt.plot(R, z, label=label, linewidth=linewidth)
    # plt.xlabel('R')
    # plt.ylabel('z')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    #
    # plt.figure(5)
    # E = 0.5 * v_norm ** 2.0
    # plt.plot(t, E, label='$E$', linewidth=linewidth)
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('kinetic energy')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure(7)
    # plt.plot(t, energy_change, linewidth=linewidth)
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('E change %')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure(8)
    # plt.plot(vz, vt, linewidth=linewidth)
    # plt.legend()
    # plt.xlabel('$v_z$')
    # plt.ylabel('$v_t$')
    # plt.grid(True)
    # plt.tight_layout()

    # if 'fig' in locals():
    #     plt.figure(6)
    # else:
    #     fig = plt.figure(6)
    #     # ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    # ax.axes.xaxis.labelpad = 5
    # ax.axes.yaxis.labelpad = 5
    # ax.axes.zaxis.labelpad = 5
    # ax.view_init(elev=30, azim=50)
    #
    # ## plot path with changing color as time evolves
    # for i in range(len(x) - 1):
    #     ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=plt.cm.jet(int(255 * i / len(x))))
    # # ax.plot(x, y, z, label=label, linewidth=linewidth, alpha=1)
    #
    # # ax.set_xlabel('x')
    # # ax.set_ylabel('y')
    # # ax.set_zlabel('z')
    # ax.set_xlabel('x/$r_{cyc}$')
    # ax.set_ylabel('y/$r_{cyc}$')
    # # ax.set_xlabel('x/l')
    # # ax.set_ylabel('y/l')
    # # ax.set_zlabel('z/l', rotation=90)
    # # ax.set_zlabel('z/$r_{cyc}$', rotation=90)
    # ax.set_zlabel('z [m]', rotation=90)
    #
    # # ax.set_xlim([-3, 3])
    # # ax.set_ylim([-3, 3])
    # # ax.set_zlim([0.5, 1.5])
    # # ax.set_zlim([-0.5, 1.5])
    # # ax.set_title('particle 3d trajectory')
    # # plt.legend()
    # plt.tight_layout()
    # # plt.tight_layout(h_pad=0.05, w_pad=0.05)

    # fig.colorbar(p[0])
    # plt.colorbar()

# p = ax.scatter([], cmap=cm.get_cmap('jet'))
# p = ax.scatter(0,0,0, alpha=1, cmap=cm.get_cmap('jet'))
# fig.colorbar(p)
# colors_list = [plt.cm.jet(int(255*i/len(x))) for i in range(len(x)-1)]
# matplotlib.colors.ListedColormap(colors_list)

# viridis = cm.get_cmap('viridis', 12)
# jetcm = cm.get_cmap('jet')
# print(viridis)

### saving figure
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/research_proposal/pics/'
# # file_name = 'mirror_trajectories_examples'
# file_name = 'mirror_trajectories_examples_with_RF'
# beingsaved = plt.figure(6)
# # beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
# beingsaved.savefig(save_dir + file_name + '.pdf', format='pdf')
