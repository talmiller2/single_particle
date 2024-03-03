import matplotlib.pyplot as plt
import numpy as np

from em_fields.RF_field_forms import E_RF_function, B_RF_function
from em_fields.default_settings import define_default_settings, define_default_field
from em_fields.em_functions import evolve_particle_in_em_fields

# from mpl_toolkits.mplot3d import Axes3D
# Axes3D = Axes3D  # pycharm auto import

plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})

plt.close('all')

# define the 3d plot
fig = plt.figure(1, figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.axes.xaxis.labelpad = 5
ax.axes.yaxis.labelpad = 5
ax.axes.zaxis.labelpad = 5
ax.view_init(elev=15, azim=-50)

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

if plot_magnetic_field_lines:
    ## plot mirror magnetic field lines
    xy_angles = np.linspace(0, 2 * np.pi, 50)
    xy_angles = xy_angles[0:-1]
    for xy_angle in xy_angles:
        # r_ini = 2
        r_ini = 20
        z_ini = settings['l'] * 0.25
        z_fin = settings['l'] * 1.5
        x_ini = [-r_ini * cyclotron_radius * np.cos(xy_angle),
                 r_ini * cyclotron_radius * np.sin(xy_angle),
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
        x_field_line = x_array[:, 0] / cyclotron_radius
        y_field_line = x_array[:, 1] / cyclotron_radius
        z_field_line = x_array[:, 2] / settings['l']
        ax.plot(x_field_line, y_field_line, z_field_line, color='k', linewidth=1, alpha=0.3)

    # add the axis line
    z_axis_line = np.array([z_ini, z_fin])
    ax.plot(0 * z_axis_line, 0 * z_axis_line, z_axis_line, color='k', linewidth=3, alpha=0.8)

# for ind_sim in range(1):
for ind_sim in range(2):

    settings = {}
    settings['trajectory_save_method'] = 'intervals'
    settings['num_snapshots'] = 300
    settings['l'] = 1.0  # m (MM cell size) # default
    # settings['l'] = 3.0  # m (MM cell size)
    settings['T_keV'] = 10.0
    # settings['T_keV'] = 30.0 / 1e3
    settings = define_default_settings(settings)

    field_dict = {}
    field_dict['mirror_field_type'] = mirror_field_type

    # field_dict['B0'] = 0.1  # Tesla (1000 Gauss)
    field_dict['B0'] = 1.0  # Tesla

    field_dict['Rm'] = 3.0  # mirror ratio
    # field_dict['Rm'] = 5.0  # mirror ratio

    field_dict['RF_type'] = 'electric_transverse'
    # field_dict['E_RF_kVm'] = 0
    # field_dict['E_RF_kVm'] = 1e-3
    # field_dict['E_RF_kVm'] = 0.1
    # field_dict['E_RF_kVm'] = 1
    field_dict['E_RF_kVm'] = 10
    # field_dict['E_RF_kVm'] = 20
    # field_dict['E_RF_kVm'] = 100
    # field_dict['RF_type'] = 'magnetic_transverse'
    # field_dict['B_RF'] = 0
    # field_dict['B_RF'] = 0.001
    # field_dict['B_RF'] = 0.01
    # field_dict['B_RF'] = 0.03
    field_dict['B_RF'] = 0.04
    # field_dict['B_RF'] = 0.05
    # field_dict['B_RF'] = 0.1
    # field_dict['lambda_RF_list'] = [1.0] # [m]
    # field_dict['lambda_RF_list'] = [-0.8] # [m]
    # field_dict['lambda_RF_list'] = [2.0] # [m]
    # field_dict['lambda_RF_list'] = [0.5] # [m]
    field_dict['lambda_RF_list'] = [10.0]  # [m]
    # field_dict['lambda_RF_list'] = [100.0] # [m]
    field_dict['alpha_RF_list'] = [1.0]
    # field_dict['alpha_RF_list'] = [1.1]
    # field_dict['alpha_RF_list'] = [0.9]
    # field_dict['alpha_RF_list'] = [0.6]

    field_dict['induced_fields_factor'] = 1.0  # default
    # field_dict['induced_fields_factor'] = 0.1
    # field_dict['induced_fields_factor'] = 0

    field_dict['with_RF_xy_corrections'] = True  # default
    # field_dict['with_RF_xy_corrections'] = False

    field_dict = define_default_field(settings, field_dict)

    loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
    # if ind_sim == 0:
    #     angle = 0.99 * loss_cone_angle
    # elif ind_sim == 1:
    #     angle = 1.01 * loss_cone_angle
    if ind_sim == 0:
        angle = 0.8 * loss_cone_angle
    elif ind_sim == 1:
        angle = 1.2 * loss_cone_angle
    x = 0
    y = np.sin(angle / 360 * 2 * np.pi)
    z = np.cos(angle / 360 * 2 * np.pi)
    unit_vec = np.array([x, y, z]).T
    v_0 = settings['v_th'] * unit_vec

    v_perp = np.sqrt(v_0[0] ** 2 + v_0[1] ** 2)

    if ind_sim == 0:
        x_0 = np.array([0, r_ini * cyclotron_radius, settings['l'] / 2.0])
    elif ind_sim == 1:
        x_0 = np.array([-r_ini * cyclotron_radius, 0, settings['l'] / 2.0])
        v_0[1] *= -1

    dt = field_dict['tau_cyclotron'] / settings['time_step_tau_cyclotron_divisions']
    # sim_cyclotron_periods = 50
    # sim_cyclotron_periods = 70
    # sim_cyclotron_periods = 100
    # tmax_mirror_lengths = 0.1
    # tmax_mirror_lengths = 1
    # tmax_mirror_lengths = 2
    tmax_mirror_lengths = 4
    sim_cyclotron_periods = int(
        tmax_mirror_lengths * settings['l'] / settings['v_th'] / field_dict['tau_cyclotron'])
    settings['sim_cyclotron_periods'] = sim_cyclotron_periods

    t_max = sim_cyclotron_periods * field_dict['tau_cyclotron']
    num_steps = int(t_max / dt)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_RF_function, B_RF_function,
                                        num_steps=num_steps, q=settings['q'], m=settings['mi'], field_dict=field_dict)
    t = hist['t']
    x = hist['x'][:, 0] / cyclotron_radius
    y = hist['x'][:, 1] / cyclotron_radius
    # x = hist['x'][:, 0] / settings['l']
    # y = hist['x'][:, 1] / settings['l']
    z = hist['x'][:, 2] / settings['l']
    # vx = hist['v'][:, 0]
    # vy = hist['v'][:, 1]
    # vz = hist['v'][:, 2]
    v_abs = np.sqrt(hist['v'][:, 0] ** 2 + hist['v'][:, 1] ** 2 + hist['v'][:, 2] ** 2)
    energy_change = 100.0 * (v_abs - v_abs[0]) / v_abs[0]

    # R = np.sqrt(x ** 2 + y ** 2)
    # v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    ### Plots
    label = '$E_{RF}$=' + str(field_dict['E_RF_kVm'])
    linewidth = 2

    # plt.figure(1)
    # # plt.subplot(1,2,1)
    # plt.plot(t, x, label='x', linewidth=linewidth, color='b')
    # plt.plot(t, y, label='y', linewidth=linewidth, color='g')
    # plt.plot(t, z, label='z', linewidth=linewidth, color='r')
    # plt.plot(t, R, label='R', linewidth=linewidth, color='k')
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('coordinate')
    # plt.grid(True)
    # plt.tight_layout()
    #
    # plt.figure(4)
    # # plt.subplot(1,2,2)
    # plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
    # plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
    # plt.plot(t, vz, label='$v_z$', linewidth=linewidth, color='r')
    # plt.plot(t, v_norm, label='$v_{norm}$', linewidth=linewidth, color='k')
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('velocity')
    # plt.grid(True)
    # plt.tight_layout()
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

    ## plot path with changing color as time evolves
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=plt.cm.jet(int(255 * i / len(x))))
    # ax.plot(x, y, z, label=label, linewidth=linewidth, alpha=1)

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.set_xlabel('x/$r_{cyc}$')
    ax.set_ylabel('y/$r_{cyc}$')
    # ax.set_xlabel('x/l')
    # ax.set_ylabel('y/l')
    ax.set_zlabel('z/l', rotation=90)

    # ax.set_xlim([-3, 3])
    # ax.set_ylim([-3, 3])
    # ax.set_zlim([0.5, 1.5])
    # ax.set_zlim([-0.5, 1.5])
    ax.set_zlim([z_ini, z_fin])
    # ax.set_title('particle 3d trajectory')
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
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/research_proposal/pics/'
# # file_name = 'mirror_trajectories_examples'
# file_name = 'mirror_trajectories_examples_with_RF'
# beingsaved = plt.figure(6)
# # beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
# beingsaved.savefig(save_dir + file_name + '.pdf', format='pdf')
