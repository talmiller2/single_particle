import copy

import matplotlib.pyplot as plt
import numpy as np

from em_fields.RF_field_forms import E_RF_function, B_RF_function, get_RF_angular_frequency
from em_fields.default_settings import define_default_settings, define_default_field
from em_fields.em_functions import evolve_particle_in_em_fields, get_cyclotron_angular_frequency

# from mpl_toolkits.mplot3d import Axes3D
# Axes3D = Axes3D  # pycharm auto import

# plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})

plt.close('all')

# define the 3d plot
fig = plt.figure(1, figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.axes.xaxis.labelpad = 5
ax.axes.yaxis.labelpad = 5
ax.axes.zaxis.labelpad = 5
# ax.view_init(elev=15, azim=-50)
ax.view_init(elev=45, azim=-45)

## plot the magnetic field lines of a mirror field alone
# plot_magnetic_field_lines = False
plot_magnetic_field_lines = True

mirror_field_type = 'post'
# mirror_field_type = 'logan'

## define cyclotron radius
settings = define_default_settings()
field_dict = {'mirror_field_type': mirror_field_type}
field_dict = define_default_field(settings, field_dict)
cyclotron_radius = settings['v_th'] / field_dict['omega_cyclotron']
field_dict_default = copy.deepcopy(field_dict)

if plot_magnetic_field_lines:
    ## plot mirror magnetic field lines
    xy_angles = np.linspace(0, 2 * np.pi, 50)
    xy_angles = xy_angles[0:-1]
    for xy_angle in xy_angles:
        # r_ini = 0
        r_ini = 1 * cyclotron_radius
        # r_ini = 2 * cyclotron_radius
        # r_ini = 4 * cyclotron_radius
        # r_ini = 10 * cyclotron_radius
        # r_ini = settings['l'] / 10
        z_ini = settings['l'] * 0.25
        z_fin = settings['l'] * 1.5
        x_ini = [-r_ini * np.cos(xy_angle),
                 r_ini * np.sin(xy_angle),
                 z_ini]
        x_array = [x_ini]
        dstep = 0.5 * cyclotron_radius
        num_field_line_steps = 600
        for j in range(num_field_line_steps):
            x_curr = x_array[-1]
            if x_curr[-1] > z_fin:
                break
            B_curr = B_RF_function(x_curr, 0, **field_dict)
            direction = B_curr / np.linalg.norm(B_curr)
            x_array += [x_curr + direction * dstep]
        x_array = np.array(x_array)
        # x_field_line = x_array[:, 0] / cyclotron_radius
        # y_field_line = x_array[:, 1] / cyclotron_radius
        # z_field_line = x_array[:, 2] / settings['l']
        x_field_line = x_array[:, 0]
        y_field_line = x_array[:, 1]
        z_field_line = x_array[:, 2]
        ax.plot(x_field_line, y_field_line, z_field_line, color='k', linewidth=1, alpha=0.3)

    # add the axis line
    z_axis_line = np.array([z_ini, z_fin])
    ax.plot(0 * z_axis_line, 0 * z_axis_line, z_axis_line, color='k', linewidth=3, alpha=0.8)

inds_sim = []
# inds_sim += [0]
# inds_sim += [1]
inds_sim += [2]
for ind_sim in inds_sim:

    settings = {}
    settings['trajectory_save_method'] = 'intervals'
    settings['num_snapshots'] = 300
    settings['l'] = 1.0  # m (MM cell size) # default
    # settings['l'] = 3.0  # m (MM cell size)
    settings['T_keV'] = 10.0
    # settings['T_keV'] = 30.0 / 1e3
    settings['time_step_tau_cyclotron_divisions'] = 20
    settings['stop_criterion'] = 't_max_adaptive_dt'
    # settings['stop_criterion'] = 't_max'
    settings = define_default_settings(settings)

    field_dict = {}
    field_dict['mirror_field_type'] = mirror_field_type

    # field_dict['B0'] = 0.1  # Tesla (1000 Gauss)
    field_dict['B0'] = 1.0  # Tesla

    field_dict['Rm'] = 3.0  # mirror ratio
    # field_dict['Rm'] = 5.0  # mirror ratio

    field_dict['RF_type'] = 'electric_transverse'
    # field_dict['RF_type'] = 'magnetic_transverse'
    if ind_sim == 0:
        field_dict['E_RF_kVm'] = 0
        field_dict['B_RF'] = 0
    else:
        # field_dict['E_RF_kVm'] = 1e-3
        # field_dict['E_RF_kVm'] = 0.1
        # field_dict['E_RF_kVm'] = 1
        # field_dict['E_RF_kVm'] = 10
        # field_dict['E_RF_kVm'] = 20
        field_dict['E_RF_kVm'] = 30
        # field_dict['E_RF_kVm'] = 50
        # field_dict['E_RF_kVm'] = 20
        # field_dict['E_RF_kVm'] = 100
        # field_dict['B_RF'] = 0
        # field_dict['B_RF'] = 0.001
        # field_dict['B_RF'] = 0.01
        # field_dict['B_RF'] = 0.03
        field_dict['B_RF'] = 0.04
        # field_dict['B_RF'] = 0.05
        # field_dict['B_RF'] = 0.1

    # field_dict['lambda_RF_list'] = [100.0] # [m]
    field_dict['alpha_RF_list'] = [1.0]
    # field_dict['alpha_RF_list'] = [1.1]
    # field_dict['alpha_RF_list'] = [0.9]
    # field_dict['alpha_RF_list'] = [0.6]
    # field_dict['alpha_RF_list'] = [1.5]

    field_dict['beta_RF_list'] = [0]
    # field_dict['beta_RF_list'] = [2]
    # field_dict['beta_RF_list'] = [6.67]

    field_dict['induced_fields_factor'] = 1.0  # default
    # field_dict['induced_fields_factor'] = 0.5
    # field_dict['induced_fields_factor'] = 0.1
    # field_dict['induced_fields_factor'] = 0

    field_dict['with_kr_correction'] = True  # default
    # field_dict['with_kr_correction'] = False
    # if ind_sim == 0:
    #     field_dict['with_kr_correction'] = False
    # elif ind_sim == 1:
    #     field_dict['with_kr_correction'] = True

    # mirror slope and RF chirp
    if ind_sim == 2:
        field_dict['use_mirror_slope'] = True
        field_dict['B_slope'] = 0.2  # [T]
        # field_dict['B_slope'] = 0.5 # [T]
        # field_dict['B_slope'] = 1.0 # [T]
        field_dict['B_slope_smooth_length'] = 0.2
        field_dict['use_RF_chirp'] = True
        # field_dict['RF_chirp_period'] = 0.9e-6 # [s]
        # field_dict['RF_chirp_period'] = 1.0 * settings['l'] / settings['v_th'] # [s]
        field_dict['RF_chirp_period'] = 1.0 * settings['l'] / settings['v_th']  # [s]
        # omega_cyc_0 = get_cyclotron_angular_frequency(settings['q'], field_dict['B0'], settings['mi'])
        field_dict['RF_chirp_amplitude'] = 0.3
        # field_dict['RF_chirp_amplitude'] = 0.9
        field_dict['RF_chirp_time_offset'] = field_dict['RF_chirp_period'] / 2

    field_dict['phase_RF_addition'] = 0
    # field_dict['phase_RF_addition'] = np.pi / 2
    # field_dict['phase_RF_addition'] = 3.6999631682413687

    field_dict = define_default_field(settings, field_dict)

    loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
    # if ind_sim == 0:
    #     angle = 0.99 * loss_cone_angle
    # elif ind_sim == 1:
    #     angle = 1.01 * loss_cone_angle
    # angle = 1.2 * loss_cone_angle
    # if ind_sim == 0:
    #     angle = 0.8 * loss_cone_angle
    # else:
    #     angle = 1.2 * loss_cone_angle
    angle = 0.8 * loss_cone_angle
    # angle = 0.5 * loss_cone_angle
    # angle = 0.1 * loss_cone_angle
    x = 0
    y = np.sin(angle / 360 * 2 * np.pi)
    z = np.cos(angle / 360 * 2 * np.pi)
    unit_vec = np.array([x, y, z]).T
    v_0 = settings['v_th'] * unit_vec
    # v_perp = np.sqrt(v_0[0] ** 2 + v_0[1] ** 2)

    x_0 = np.array([0, r_ini, settings['l'] / 2.0])
    # if ind_sim == 0:
    #     x_0 = np.array([0, r_ini, settings['l'] / 2.0])
    # elif ind_sim == 1:
    #     x_0 = np.array([-r_ini, 0, settings['l'] / 2.0])
    #     v_0[1] *= -1
    # elif ind_sim == 2:
    #     # testing exploding fields case
    #     x_0 = np.array([0., 0., 0.5])
    #     v_0 = np.array([1270241.4443987, 1084444.96273159, 155314.40349702])

    dt = field_dict['tau_cyclotron'] / settings['time_step_tau_cyclotron_divisions']
    # sim_cyclotron_periods = 50
    # sim_cyclotron_periods = 70
    # sim_cyclotron_periods = 100
    # tmax_mirror_lengths = 0.1
    # tmax_mirror_lengths = 1
    # tmax_mirror_lengths = 3
    tmax_mirror_lengths = 4
    # tmax_mirror_lengths = 5
    sim_cyclotron_periods = int(
        tmax_mirror_lengths * settings['l'] / settings['v_th'] / field_dict['tau_cyclotron'])
    settings['sim_cyclotron_periods'] = sim_cyclotron_periods

    t_max = sim_cyclotron_periods * field_dict['tau_cyclotron']
    num_steps = int(t_max / dt)
    # num_steps = 1000
    num_steps = int(1e15)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_RF_function, B_RF_function,
                                        stop_criterion=settings['stop_criterion'], num_steps=num_steps, t_max=t_max,
                                        q=settings['q'], m=settings['mi'], field_dict=field_dict,
                                        r_max=settings['l'],
                                        )
    t = hist['t']
    x = hist['x'][:, 0]
    y = hist['x'][:, 1]
    # x = hist['x'][:, 0] / cyclotron_radius
    # y = hist['x'][:, 1] / cyclotron_radius
    # x = hist['x'][:, 0] / settings['l']
    # y = hist['x'][:, 1] / settings['l']
    z = hist['x'][:, 2]
    # z = hist['x'][:, 2] / settings['l']
    # vx = hist['v'][:, 0]
    # vy = hist['v'][:, 1]
    # vz = hist['v'][:, 2]
    # v_abs = np.sqrt(hist['v'][:, 0] ** 2 + hist['v'][:, 1] ** 2 + hist['v'][:, 2] ** 2)
    # energy_change = 100.0 * ((v_abs / v_abs[0]) ** 2 - 1)
    E_tot = hist['v'][:, 0] ** 2 + hist['v'][:, 1] ** 2 + hist['v'][:, 2] ** 2
    E_transverse = hist['v'][:, 0] ** 2 + hist['v'][:, 1] ** 2
    energy_transverse_ratio = E_transverse / E_tot

    # mapping to center of mirror
    vt = np.sqrt(hist['v'][:, 0] ** 2 + hist['v'][:, 1] ** 2)
    vz = hist['v'][:, 2]
    Bz = hist['B'][:, 2]
    vt_adj = vt * np.sqrt(Bz[0] / Bz)
    det = vz ** 2.0 + vt ** 2.0 * (1 - Bz[0] / Bz)
    vz_adj = np.sign(vz) * np.sqrt(det)
    energy_transverse_ratio_adj = vt_adj ** 2 / (vt_adj ** 2 + vz_adj ** 2)

    R = np.sqrt(x ** 2 + y ** 2)
    # v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    omega_cyc_0 = get_cyclotron_angular_frequency(settings['q'], field_dict['B0'], settings['mi'])
    omega_cyc_local = get_cyclotron_angular_frequency(settings['q'], Bz, settings['mi'])
    omega_RF_base = field_dict['omega_RF'][0]
    omega_RF_chirped = get_RF_angular_frequency(omega_RF_base, t, field_dict)

    ### Plots
    # t_label = 't [s]'
    t /= field_dict['tau_cyclotron']
    t_label = '$t / \\tau_{cyc,0}$'

    label = '$E_{RF}$=' + str(field_dict['E_RF_kVm'])
    linewidth = 2

    if ind_sim == inds_sim[0]:
        # plt.figure(2, figsize=(14, 5))
        plt.figure(2, figsize=(12, 9))

    if field_dict['RF_type'] == 'electric_transverse':
        RF_type_short = 'REF'
    else:
        RF_type_short = 'RMF'

    if ind_sim == 0:
        # linestyle = '-.'
        linestyle = '-'
        color = 'g'
        sim_label = 'w/o RF'
    elif ind_sim == 1:
        # linestyle = '--'
        linestyle = '-'
        color = 'b'
        sim_label = RF_type_short
    else:
        linestyle = '-'
        sim_label = RF_type_short + '+slope+chirp'
        color = 'r'

    # plt.subplot(1, 3, 1)
    plt.subplot(2, 2, 1)
    # plt.plot(t, x, label='x', linewidth=linewidth, color='b', linestyle=linestyle)
    # plt.plot(t, y, label='y', linewidth=linewidth, color='g', linestyle=linestyle)
    # plt.plot(t, z, linewidth=linewidth, color='r', linestyle=linestyle, label='z ' + sim_label)
    plt.plot(t, z, linewidth=linewidth, color=color, linestyle=linestyle, label=sim_label)
    # plt.plot(t, R, linewidth=linewidth, color='k', linestyle=linestyle, label='R ' + sim_label)
    plt.legend()
    plt.xlabel(t_label)
    # plt.ylabel('coordinate [m]')
    plt.ylabel('z [m]')
    # plt.title('ind_sim = ' + str(ind_sim))
    plt.grid(True)
    plt.tight_layout()
    #
    # plt.figure(4)
    # # plt.subplot(1,2,2)
    # plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
    # plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
    # plt.plot(t, vz, label='$v_z$', linewidth=linewidth, color='r')
    # plt.plot(t, v_norm, label='$v_{norm}$', linewidth=linewidth, color='k')
    # plt.legend()
    # plt.x_label('t')
    # plt.y_label('velocity')
    # plt.grid(True)
    # plt.tight_layout()
    #
    # plt.figure(2)
    # plt.plot(x, y, linewidth=linewidth)
    # plt.x_label('x')
    # plt.y_label('y')
    # plt.grid(True)
    # plt.tight_layout()
    #
    # plt.figure(3)
    # plt.plot(R, z, label=label, linewidth=linewidth)
    # plt.x_label('R')
    # plt.y_label('z')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    #
    # plt.figure(5)
    # E = 0.5 * v_norm ** 2.0
    # plt.plot(t, E, label='$E$', linewidth=linewidth)
    # plt.legend()
    # plt.x_label('t')
    # plt.y_label('kinetic energy')
    # plt.grid(True)
    # plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.plot(t, E_tot / E_tot[0], label=sim_label, linestyle=linestyle, linewidth=linewidth, color=color)
    plt.legend()
    plt.xlabel(t_label)
    plt.ylabel('$E/E_0$')
    plt.grid(True)
    plt.tight_layout()

    # plt.figure(7)
    # plt.subplot(1, 3, 2)
    plt.subplot(2, 2, 3)
    if ind_sim == inds_sim[0]:
        plt.plot(t, 0 * t + 1 / field_dict['Rm'], linewidth=linewidth, linestyle='--', color='k', label='$1/R_m$')
    # plt.plot(t, energy_change, linewidth=linewidth)
    # plt.plot(t, energy_transverse_ratio, linewidth=linewidth, linestyle=linestyle, color='b')
    plt.plot(t, energy_transverse_ratio_adj, linewidth=linewidth, linestyle=linestyle, color=color, label=sim_label)
    plt.legend()
    plt.xlabel(t_label)
    # plt.ylabel('E change %')
    plt.ylabel('adjusted $E_{\perp}/E_{tot}$')
    plt.grid(True)
    plt.tight_layout()

    # # plt.figure(8)
    # plt.subplot(1, 3, 2)
    # plt.plot(t, hist['B'][:, 0], label='$B_x$', linewidth=linewidth, color='b')
    # plt.plot(t, hist['B'][:, 1], label='$B_y$', linewidth=linewidth, color='g')
    # plt.plot(t, hist['B'][:, 2], label='$B_z$', linewidth=linewidth, color='r')
    # plt.plot(t, np.linalg.norm(hist['B'], axis=1), label='$B_{norm}$', linewidth=linewidth, color='k')
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('B')
    # plt.grid(True)
    # plt.tight_layout()

    # # plt.figure(9)
    # plt.subplot(1, 3, 3)
    # plt.plot(t, hist['dt'], linewidth=linewidth, color='b')
    # plt.ylim([0, 1.1 * max(hist['dt'])])
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('dt')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure(9)
    # plt.subplot(1, 3, 3)
    plt.subplot(2, 2, 4)
    # plt.plot(t, omega_cyc_local / omega_cyc_0, linewidth=linewidth, linestyle=linestyle, color='b', label='$\\omega_{cyc}$ ' + sim_label)
    # plt.plot(t, omega_RF_chirped / omega_cyc_0, linewidth=linewidth, linestyle=linestyle, color='r', label='$\\omega_{RF}$ ' + sim_label)
    plt.plot(t, omega_cyc_local / omega_cyc_0, linewidth=linewidth, linestyle='-', color=color,
             label='$\\omega_{cyc}$ ' + sim_label)
    plt.plot(t, omega_RF_chirped / omega_cyc_0, linewidth=linewidth, linestyle='--', color=color,
             label='$\\omega_{RF}$ ' + sim_label)
    plt.legend()
    plt.xlabel(t_label)
    # plt.ylabel('$\\omega$')
    plt.ylabel('$\\omega / \\omega_{cyc,0}$')
    plt.grid(True)
    plt.tight_layout()

    ## plot path with changing color as time evolves
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=plt.cm.jet(int(255 * i / len(x))))
    # ax.plot(x, y, z, label=label, linewidth=linewidth, alpha=1)

    # ax.set_xlabel('x/$r_{cyc}$')
    # ax.set_ylabel('y/$r_{cyc}$')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # ax.set_xlabel('x/l')
    # ax.set_ylabel('y/l')
    # ax.set_zlabel('z/l', rotation=90)
    ax.set_zlabel('z [m]', rotation=90)

    # ax.set_xlim([-3, 3])
    # ax.set_ylim([-3, 3])
    ax.set_xlim([-2 * r_ini, 2 * r_ini])
    ax.set_ylim([-2 * r_ini, 2 * r_ini])
    # ax.set_zlim([0.5, 1.5])
    # ax.set_zlim([-0.5, 1.5])
    ax.set_zlim([z_ini, z_fin])
    # ax.set_title('particle 3d trajectory')
    ax.set_title('particle 3d trajectory: ' + sim_label)
    # plt.legend()
    plt.tight_layout()
    # plt.tight_layout(h_pad=0.05, w_pad=0.05)

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
save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/research_update/pics/'

# file_name = 'mirror_trajectories_metrics'
# beingsaved = plt.figure(2)
# beingsaved.savefig(save_dir + file_name + '.pdf', format='pdf')

# file_name = 'mirror_trajectories_examples_noRF'
# file_name = 'mirror_trajectories_examples_REF'
# file_name = 'mirror_trajectories_examples_REF_slope_chirp'
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.pdf', format='pdf')
