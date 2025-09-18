import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from em_fields.MMM_field_forms import get_MMM_electric_field, get_MMM_magnetic_field
from em_fields.default_settings import define_default_settings, define_default_field
from em_fields.em_functions import evolve_particle_in_em_fields

# from mpl_toolkits.mplot3d import Axes3D
# Axes3D = Axes3D  # pycharm auto import

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})

plt.close('all')

plot_1d = False
plot_3d = False
if plot_3d:
    # define the 3d plot
    fig = plt.figure(1, figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.axes.xaxis.labelpad = 5
    ax.axes.yaxis.labelpad = 5
    ax.axes.zaxis.labelpad = 5
    ax.view_init(elev=15, azim=-50)

settings = {}
# settings['time_step_tau_cyclotron_divisions'] = 5
settings['time_step_tau_cyclotron_divisions'] = 20
settings['stop_criterion'] = 't_max_adaptive_dt'
settings = define_default_settings(settings)

field_dict = {}
# field_dict['use_static_main_cell'] = True
field_dict['use_static_main_cell'] = False
field_dict['Rm'] = 6
# field_dict['Rm'] = 5
# field_dict['Rm'] = 3
field_dict['Rm_main'] = 3
field_dict['MMM_z_wall'] = 1.0  # [m]
# field_dict['MMM_z_wall'] = 1.2  # [m]
# field_dict['MMM_z_wall'] = 0.8  # [m]
# field_dict['U_MMM'] = 0
# field_dict['U_MMM'] = 1e-4 * settings['v_th']
# field_dict['U_MMM'] = 0.01 * settings['v_th']
# field_dict['U_MMM'] = 0.05 * settings['v_th']
field_dict['U_MMM'] = 0.1 * settings['v_th']
# field_dict['U_MMM'] = 0.5 * settings['v_th']
# field_dict['U_MMM'] = 1.0 * settings['v_th']
# field_dict['induced_fields_factor'] = 0
field_dict['induced_fields_factor'] = 1
field_dict = define_default_field(settings, field_dict)
# tau = settings['l'] / settings['v_th']
# tau = settings['l'] / field_dict['U_MMM']

cyclotron_radius = settings['v_th'] / field_dict['omega_cyclotron']

# loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
loss_cone_angle = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm_main']))

def plot_MMM_lines(t, t_fac, plot_static_cell):
    num_lines = 10
    for sign in [+1, -1]:
        for i in range(num_lines):
            z = field_dict['z_mirror_shift'] + i * field_dict['l'] - field_dict['U_MMM'] * t
            z *= sign
            ind_wall_first = np.where(abs(z) < field_dict['MMM_z_wall'])[0]
            if len(ind_wall_first) > 0:
                z[ind_wall_first[0]:] = np.nan
            if sign == 1 and i == 0:
                # label = '$B_{max}$'
                label = '$B_{max}$ lines'
            else:
                label = None
            plt.plot(t * t_fac, z, linewidth=2, color='grey', alpha=0.5, label=label)

        if plot_static_cell:
            plt.plot(t * t_fac, sign * field_dict['MMM_static_main_cell_z'] + 0 * t, linewidth=2, color='grey',
                     alpha=0.7)

    return


# use_random_particles = False
use_random_particles = True

initialize_inside_MMM = False
# initialize_inside_MMM = True

save_figures = False
# save_figures = True


if use_random_particles:
    # generate several particles that are in the static LC, with general angle
    num_particles = 10
    inds_sim = range(num_particles)

    np.random.seed(0)
    theta_angles = np.random.uniform(loss_cone_angle, 180 - loss_cone_angle, num_particles)
    phi_angles = np.random.uniform(0, 360, num_particles)

    x = np.sin(theta_angles / 360 * 2 * np.pi) * np.cos(phi_angles / 360 * 2 * np.pi)
    y = np.sin(theta_angles / 360 * 2 * np.pi) * np.sin(phi_angles / 360 * 2 * np.pi)
    z = np.cos(theta_angles / 360 * 2 * np.pi)
    unit_vecs = np.array([x, y, z]).T

    # # check units vectors are indeed unit
    # for i in range(unit_vecs.shape[0]):
    #     print(np.linalg.norm(unit_vecs[i, :]))

else:
    inds_sim = []
    # inds_sim += [0]
    # inds_sim += [1]
    inds_sim += [2]
    # inds_sim += [3]
    # inds_sim += [4]
    # inds_sim += [5]
    # inds_sim += [6]
    # inds_sim += [7]

t_list, x_list, y_list, R_list, z_list, v_abs_list, dE_list = [], [], [], [], [], [], []

for ind_sim in inds_sim:
    print(f'ind_sim={ind_sim}')
    # if ind_sim == 0:
    # angle = 0.99 * loss_cone_angle
    # angle = 1.05 * loss_cone_angle
    # angle = 1.2 * loss_cone_angle
    # angle = 1.5 * loss_cone_angle
    # angle = 0.99 * theta_high
    # angle = 1.01 * theta_high
    # angle = 1.1 * theta_high
    # angle = 1.5 * theta_high
    # angle = 1.01 * theta_low
    # if ind_sim <= 5:
    #     angle = 60
    # else:
    #     angle = 80

    # angle = 0.99 * loss_cone_angle    # elif ind_sim == 1:
    #     angle = 1.01 * loss_cone_angle
    # angle = 1.2 * loss_cone_angle
    # if ind_sim == 0:
    #     angle = 0.8 * loss_cone_angle
    # else:
    #     angle = 1.2 * loss_cone_angle
    # x = 0
    # y = np.sin(angle / 360 * 2 * np.pi)
    # z = np.cos(angle / 360 * 2 * np.pi)
    # unit_vec = np.array([x, y, z]).T
    # v_0 = settings['v_th'] * unit_vec
    # v_perp = np.sqrt(v_0[0] ** 2 + v_0[1] ** 2)

    v_0 = settings['v_th'] * unit_vecs[ind_sim]

    if initialize_inside_MMM:
        v_0[2] -= field_dict['U_MMM']

    r_ini = 0
    # r_ini = 1 * cyclotron_radius
    # r_ini = 5 * cyclotron_radius
    # if ind_sim <= 1:
    #     z_ini = 0
    # # elif ind_sim in [2, 3]:
    # elif ind_sim in [2, 3, 6, 7]:
    #     z_ini = 3.5
    # elif ind_sim in [4, 5]:
    #     z_ini = -3.5

    z_ini = 0
    if initialize_inside_MMM:
        z_ini = 3.5


    # z_ini = 2
    # x_0 = np.array([0, r_ini, settings['l'] / 2.0])
    # if ind_sim == 0:
    #     x_0 = np.array([0, r_ini, z_ini])
    # if ind_sim == 1:
    # if ind_sim in [1, 3, 5, 7]:
    #     # x_0 = np.array([-r_ini, 0, z_ini])
    #     # v_0[1] *= -1
    #     v_0[2] *= -1

    x_0 = np.array([0, r_ini, z_ini])

    dt = field_dict['tau_cyclotron'] / settings['time_step_tau_cyclotron_divisions']
    # tmax_mirror_lengths = 10
    tmax_mirror_lengths = 50
    # tmax_mirror_lengths = 60
    sim_cyclotron_periods = int(
        tmax_mirror_lengths * settings['l'] / settings['v_th'] / field_dict['tau_cyclotron'])
    settings['sim_cyclotron_periods'] = sim_cyclotron_periods

    t_max = sim_cyclotron_periods * field_dict['tau_cyclotron']
    num_steps = int(t_max / dt)
    # num_steps = 1000
    num_steps = int(1e15)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, get_MMM_electric_field, get_MMM_magnetic_field,
                                        stop_criterion=settings['stop_criterion'], num_steps=num_steps, t_max=t_max,
                                        q=settings['q'], m=settings['mi'], field_dict=field_dict,
                                        r_max=settings['l'],
                                        )
    t = hist['t']
    x = hist['x'][:, 0]
    y = hist['x'][:, 1]
    R = np.sqrt(x ** 2 + y ** 2)
    z = hist['x'][:, 2]
    vx = hist['v'][:, 0]
    vy = hist['v'][:, 1]
    vz = hist['v'][:, 2]
    vt = np.sqrt(vx ** 2 + vy ** 2)
    v_abs = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    if initialize_inside_MMM:
        v_abs_moving = np.sqrt(vx ** 2 + vy ** 2 + (vz + field_dict['U_MMM']) ** 2)
        dE = 100.0 * (v_abs_moving - v_abs_moving[0]) / v_abs_moving[0]
    else:
        dE = 100.0 * (v_abs - v_abs[0]) / v_abs[0]

    # v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    ### Plots
    label = '$E_{RF}$=' + str(field_dict['E_RF_kVm'])
    linewidth = 2

    # t_label = '$t$ [s]'
    # t_fac = 1
    t_label = '$t \cdot v_{th} / l$'
    t_fac = settings['v_th'] / settings['l']
    v_fac = 1 / settings['v_th']

    t_list += [t]
    x_list += [x]
    y_list += [y]
    R_list += [R]
    z_list += [z]
    v_abs_list += [v_abs]
    dE_list += [dE]

    if plot_1d:
        # plt.figure(2, figsize=(14, 5))
        plt.figure(num=None, figsize=(14, 5))
        # plt.subplot(1, 3, 1)
        plt.subplot(1, 4, 1)
        plot_MMM_lines(t, t_fac, field_dict['use_static_main_cell'])
        plt.plot(t * t_fac, x, label='x', linewidth=linewidth, color='b')
        plt.plot(t * t_fac, y, label='y', linewidth=linewidth, color='g')
        plt.plot(t * t_fac, z, label='z', linewidth=linewidth, color='r')
        plt.plot(t * t_fac, R, label='R', linewidth=linewidth, color='k')
        plt.legend()
        plt.xlabel(t_label)
        plt.ylabel('coordinate [m]')
        plt.title('ind_sim = ' + str(ind_sim))
        plt.grid(True)
        plt.tight_layout()

        # plt.figure(4)
        plt.subplot(1, 4, 2)
        # plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
        # plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
        plt.plot(t * t_fac, vz * v_fac, label='$v_z$', linewidth=linewidth, color='r')
        plt.plot(t * t_fac, abs(vz) * v_fac, label='$|v_z|$', linewidth=linewidth, color='orange')
        plt.plot(t * t_fac, vt * v_fac, label='$v_{\\perp}$', linewidth=linewidth, color='k')
        plt.plot(t * t_fac, v_abs * v_fac, label='$|v|$', linewidth=linewidth, color='grey')
        plt.legend()
        plt.xlabel(t_label)
        # plt.ylabel('velocity [m/s]')
        plt.ylabel('$v / v_{th}$')
        plt.grid(True)
        plt.tight_layout()

        # plt.figure(3)
        # plt.plot(R, z, label=label,s linewidth=linewidth)
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

        # plt.figure(7)
        # plt.plot(t, energy_change, linewidth=linewidth)
        # plt.legend()
        # plt.x_label('t')
        # plt.y_label('E change %')
        # plt.grid(True)
        # plt.tight_layout()

        # plt.figure(8)
        # plt.subplot(1, 3, 2)
        plt.subplot(1, 4, 3)
        plt.plot(t * t_fac, hist['B'][:, 0], label='$B_x$', linewidth=linewidth, color='b')
        plt.plot(t * t_fac, hist['B'][:, 1], label='$B_y$', linewidth=linewidth, color='g')
        plt.plot(t * t_fac, hist['B'][:, 2], label='$B_z$', linewidth=linewidth, color='r')
        plt.plot(t * t_fac, np.linalg.norm(hist['B'], axis=1), label='$|B|$', linewidth=linewidth, color='k')
        plt.legend()
        plt.xlabel(t_label)
        plt.ylabel('B [T]')
        plt.grid(True)
        plt.tight_layout()

        # plt.figure(9)
        # plt.subplot(1, 3, 3)
        # plt.plot(t, hist['dt'], linewidth=linewidth, color='b')
        # plt.ylim([0, 1.1 * max(hist['dt'])])
        # plt.legend()
        # plt.xlabel(t_label)
        # plt.ylabel('dt')
        # plt.grid(True)
        # plt.tight_layout()

        # plt.subplot(1, 3, 3)
        plt.subplot(1, 4, 4)
        E_fac = 1e-3
        plt.plot(t * t_fac, hist['E'][:, 0] * E_fac, label='$E_x$', linewidth=linewidth, color='b')
        plt.plot(t * t_fac, hist['E'][:, 1] * E_fac, label='$E_y$', linewidth=linewidth, color='g')
        plt.plot(t * t_fac, hist['E'][:, 2] * E_fac, label='$E_z$', linewidth=linewidth, color='r')
        plt.plot(t * t_fac, np.linalg.norm(hist['E'], axis=1) * E_fac, label='$|E|$', linewidth=linewidth, color='k')
        plt.legend()
        plt.xlabel(t_label)
        plt.ylabel('E [kV/m]')
        plt.grid(True)
        plt.tight_layout()

    if plot_3d:
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
        # ax.set_zlim([z_ini, z_fin])
        # ax.set_title('particle 3d trajectory')
        # plt.legend()
        plt.tight_layout()
        # plt.tight_layout(h_pad=0.05, w_pad=0.05)

# combined trajectories plot
# plt.figure(figsize=(7, 5))
plt.figure(figsize=(14, 5))
colors = cm.rainbow(np.linspace(0, 1, len(t_list)))
plt.subplot(1, 2, 1)
plot_MMM_lines(t, t_fac, field_dict['use_static_main_cell'])
for i, (t, z, color) in enumerate(zip(t_list, z_list, colors)):
    plt.plot(t * t_fac, z,
             # label='#' + str(i + 1),
             linewidth=linewidth, color=color)
plt.legend(loc='upper right')
plt.xlabel(t_label)
plt.ylabel('z [m]')

title = f"Fields MMM: $z_{{wall}}=${field_dict['MMM_z_wall']}m, $R_m=${field_dict['Rm']}"
if field_dict['use_static_main_cell']:
    title += f", static main cell: $z_{{main}}=${field_dict['MMM_static_main_cell_z']}m, $R_m=${field_dict['Rm_main']}"
if field_dict['induced_fields_factor'] == 0:
    title += ', w/o E fields'
# plt.title(title)
plt.suptitle(title)
plt.ylim([-5, 5])
# plt.grid(True)
# plt.tight_layout()

# combined dE plot
# plt.figure(num=None, figsize=(7, 5))
plt.subplot(1, 2, 2)
colors = cm.rainbow(np.linspace(0, 1, len(t_list)))
for i, (t, z, dE, color) in enumerate(zip(t_list, z_list, dE_list, colors)):
    plt.plot(t * t_fac, dE,
             label='#' + str(i + 1),
             linewidth=linewidth, color=color)
plt.legend(loc='upper left')
plt.xlabel(t_label)
if initialize_inside_MMM:
    plt.ylabel('$\\Delta E$ [%] (in MMM reference frame)')
else:
    plt.ylabel('$\\Delta E$ [%] (in lab frame)')
# title = 'main cell: $z_{main}=$' + str(field_dict['MMM_static_main_cell_z']) + 'm, $R_m=$' + str(field_dict['Rm_main'])
# title += ', MMM: $z_{wall}=$' + str(field_dict['MMM_z_wall']) + 'm, $R_m=$' + str(field_dict['Rm'])
# if field_dict['induced_fields_factor'] == False:
#     title += ', w/o E fields'
# plt.title(title)
plt.grid(True)
plt.tight_layout()

### saving figures
if save_figures:
    fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2026/pics/'
    file_name = 'single_particle_trajectories_MMM'
    if initialize_inside_MMM:
        file_name += '_init_in_MMM'
    else:
        file_name += '_init_in_maincell'
    if field_dict['use_static_main_cell']:
        file_name += '_with_static_maincell'
    if field_dict['induced_fields_factor'] == 0:
        file_name += '_withoutE'
    plt.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)

# # combined x,y plot
# plt.figure(num=None, figsize=(7, 5))
# colors = cm.rainbow(np.linspace(0, 1, len(t_list)))
# for i, (t, x, y, color) in enumerate(zip(t_list, x_list, y_list, colors)):
#     plt.plot(t * t_fac, x, label='#' + str(i), linewidth=linewidth, color=color)
#     plt.plot(t * t_fac, y, linestyle='--', linewidth=linewidth, color=color)
# plt.legend()
# plt.xlabel(t_label)
# plt.ylabel('x, y [m]')
# title = 'main cell: $z_{main}=$' + str(field_dict['MMM_static_main_cell_z']) + 'm, $R_m=$' + str(field_dict['Rm_main'])
# title += ', MMM: $z_{wall}=$' + str(field_dict['MMM_z_wall']) + 'm, $R_m=$' + str(field_dict['Rm'])
# if field_dict['induced_fields_factor'] == False:
#     title += ', w/o E fields'
# plt.title(title)
# # plt.ylim([-5, 5])
# plt.grid(True)
# plt.tight_layout()
#
#
# # combined R plot
# plt.figure(num=None, figsize=(7, 5))
# colors = cm.rainbow(np.linspace(0, 1, len(t_list)))
# for i, (t, R, color) in enumerate(zip(t_list, R_list, colors)):
#     plt.plot(t * t_fac, R, label='#' + str(i), linewidth=linewidth, color=color)
# plt.legend()
# plt.xlabel(t_label)
# plt.ylabel('R [m]')
# title = 'main cell: $z_{main}=$' + str(field_dict['MMM_static_main_cell_z']) + 'm, $R_m=$' + str(field_dict['Rm_main'])
# title += ', MMM: $z_{wall}=$' + str(field_dict['MMM_z_wall']) + 'm, $R_m=$' + str(field_dict['Rm'])
# if field_dict['induced_fields_factor'] == False:
#     title += ', w/o E fields'
# plt.title(title)
# # plt.ylim([-5, 5])
# plt.grid(True)
# plt.tight_layout()


# # combined velocity plot
# plt.figure(num=None, figsize=(7, 5))
# colors = cm.rainbow(np.linspace(0, 1, len(t_list)))
# for i, (t, z, v_abs, color) in enumerate(zip(t_list, z_list, v_abs_list, colors)):
#     plt.plot(t * t_fac, v_abs * v_fac, label='#' + str(i), linewidth=linewidth, color=color)
# plt.legend()
# plt.xlabel(t_label)
# plt.ylabel('$v / v_{th}$')
# title = 'main cell: $z_{main}=$' + str(field_dict['MMM_static_main_cell_z']) + 'm, $R_m=$' + str(field_dict['Rm_main'])
# title += ', MMM: $z_{wall}=$' + str(field_dict['MMM_z_wall']) + 'm, $R_m=$' + str(field_dict['Rm'])
# if field_dict['induced_fields_factor'] == False:
#     title += ', w/o E fields'
# plt.title(title)
# plt.grid(True)
# plt.tight_layout()
