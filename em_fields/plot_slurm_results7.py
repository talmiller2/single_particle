import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# plt.close('all')

save_dir_main = '/Users/talmiller/Downloads/single_particle/'
# save_dir_main += '/set14_T_B0_1T_l_1m_randphase_save_intervals/'
# save_dir_main += '/set15_T_B0_1T_l_1m_Logan_intervals/'
# save_dir_main += '/set16_T_B0_1T_l_1m_Post_intervals/'
save_dir_main += '/set17_T_B0_1T_l_3m_Post_intervals/'

set_names = []

Rm = 2
# Rm = 4

# ERF = 0
# ERF = 1
# ERF = 5
ERF = 10
# ERF = 30
# ERF = 100

# alpha = 0.6
# alpha = 0.8
# alpha = 1.0
# alpha = 1.2
# alpha = 1.5
alpha = 2.0
# alpha = 2.5
# alpha = 3.0

# vz_res = 0.5
# vz_res = 1.0
# vz_res = 1.5
vz_res = 2.0
# vz_res = 2.5
# vz_res = 3.0

# color = 'b'
color = 'g'
# color = 'r'
# color = 'm'

omega_RF_over_omega_cyc_0 = alpha
v_RF = vz_res * alpha / (alpha + 1e-3 - 1.0)
print('vz_res/v_th = ' + str(vz_res) + ', alpha = ' + str(alpha))
print('omega_RF/omega_cyc0 = ' + '{:.2f}'.format(omega_RF_over_omega_cyc_0) + ', v_RF/v_th = ' + '{:.2f}'.format(v_RF))

if ERF > 0:
    set_names += ['Rm_' + str(Rm) + '_ERF_' + str(ERF) + '_alpha_' + str(alpha) + '_vz_' + str(vz_res)]
else:
    set_names += ['Rm_' + str(Rm) + '_ERF_0']

for set_ind in range(len(set_names)):
    set_name = set_names[set_ind]
    save_dir = save_dir_main + set_name

    # load runs data
    data_dict_file = save_dir + '.pickle'
    with open(data_dict_file, 'rb') as fid:
        data_dict = pickle.load(fid)
    settings = data_dict['settings']
    field_dict = data_dict['field_dict']

    # draw trajectories for several particles
    num_particles = len(data_dict['z'])
    # ind_points = [0, 1, 2, 4, 5]
    # ind_points = [0]
    # ind_points = [3]
    # ind_points = [801]
    # ind_points = [537, 752, 802]
    # ind_points = [1, 4]
    # ind_points = range(5)
    # ind_points = range(10)
    # ind_points = range(20)
    # ind_points = range(100)
    # ind_points = range(500)
    ind_points = range(1000)
    # ind_points = range(2000)
    # ind_points = range(100, 200)
    # ind_points = range(20, 30)
    # ind_points = range(30, 40)
    # ind_points = range(num_particles)

    points_counter = 0

    for ind_point in ind_points:
        # skip = 1
        # # skip = 2
        # # num_snapshots = len(data_dict['t'][ind_point])
        # t = np.array(data_dict['t'][ind_point])[0::skip]
        # z = np.array(data_dict['z'][ind_point])[0::skip]
        # v = np.array(data_dict['v'][ind_point])[0::skip]
        # v_transverse = np.array(data_dict['v_transverse'][ind_point])[0::skip]
        # v_axial = np.array(data_dict['v_axial'][ind_point])[0::skip]
        # Bz = np.array(data_dict['Bz'][ind_point])[0::skip]

        inds_trajectory = range(len(data_dict['t'][ind_point]))
        # inds_trajectory = np.argsort(data_dict['t'][ind_point])

        t = np.array(data_dict['t'][ind_point])[inds_trajectory]
        z = np.array(data_dict['z'][ind_point])[inds_trajectory]
        v = np.array(data_dict['v'][ind_point])[inds_trajectory]
        v_transverse = np.array(data_dict['v_transverse'][ind_point])[inds_trajectory]
        v_axial = np.array(data_dict['v_axial'][ind_point])[inds_trajectory]
        Bz = np.array(data_dict['Bz'][ind_point])[inds_trajectory]

        if ind_point == 0:
            counter_particles_trapped = 0 * t
            counter_particles_trapped_and_axis_bound = 0 * t
            counter_particles_trapped_or_left = 0 * t
            counter_particles_trapped_or_left_and_axis_bound = 0 * t

        # calculate if a particle is initially in right loss cone
        in_loss_cone = (v_transverse[0] / v[0]) ** 2 < 1 / field_dict['Rm']
        positive_z_velocity = v_axial[0] > 0

        if in_loss_cone and positive_z_velocity:  # right loss cone
            linestyle = '-'
            linewidth = 1
            # do_plot = True
            do_plot = False
        elif in_loss_cone and not positive_z_velocity:  # left loss cone
            # linestyle = ':'
            linestyle = '-'
            linewidth = 1
            # do_plot = True
            do_plot = False
        else:  # trapped
            # linestyle = '--'
            linestyle = '-'
            linewidth = 1
            do_plot = True
            # do_plot = False

        # plots
        if do_plot:
            points_counter += 1

            # check loss cone criterion as a function of time, for varying Bz(t)
            Rm_dynamic = field_dict['B0'] * field_dict['Rm'] / Bz
            out_of_loss_cone = (v_transverse / v) ** 2 >= 1 / Rm_dynamic
            in_loss_cone = (v_transverse / v) ** 2 < 1 / Rm_dynamic

            # z_cutoff = 10
            z_cutoff = 5.5
            z_axis_bound = z / settings['l'] <= z_cutoff
            trapped = out_of_loss_cone * z_axis_bound

            is_particle_escaping_left = in_loss_cone * (v_axial < 0)
            counter_particles_trapped += 1.0 * out_of_loss_cone
            counter_particles_trapped_and_axis_bound += 1.0 * out_of_loss_cone * z_axis_bound
            counter_particles_trapped_or_left += 1.0 * out_of_loss_cone + is_particle_escaping_left
            counter_particles_trapped_or_left_and_axis_bound += 1.0 * out_of_loss_cone * z_axis_bound + is_particle_escaping_left

            # TODO: need to make an irreversible counter

    percent_particles_trapped = counter_particles_trapped / points_counter * 100.0
    percent_particles_trapped_and_axis_bound = counter_particles_trapped_and_axis_bound / points_counter * 100.0
    percent_particles_trapped_or_left = counter_particles_trapped_or_left / points_counter * 100.0
    percent_particles_trapped_or_left_and_axis_bound = counter_particles_trapped_or_left_and_axis_bound \
                                                       / points_counter * 100.0

    # t_axis = t / field_dict['tau_cyclotron']
    t_axis = t * 1e6

    plt.figure(1)
    # plt.plot(t_axis, percent_particles_trapped, '-b', label='out of LC')
    # plt.plot(t_axis, percent_particles_trapped_and_axis_bound, '-r',
    #          label='out of LC, and z/l<' + str(z_cutoff))
    # plt.plot(t_axis, percent_going_left, '-g', label='in LC, going left')
    # color = 'b'
    label_RF_params = ', for $v_{z,res}/v_{th}$=' + str(vz_res) + ', $\\alpha=\\omega_{RF}/\\omega_{cyc0}=$' + str(
        alpha) + ', $v_{RF}/v_{th}$=' + '{:.1f}'.format(v_RF)
    # label = 'trapped or escaping left'
    label = 'trapped/left' + label_RF_params
    plt.plot(t_axis, percent_particles_trapped_or_left, '-', color=color, label=label)
    # color = 'r'
    # label = 'trapped or escaping left, and z/l<' + str(z_cutoff)
    label = 'same as last, and z/l<' + str(z_cutoff)
    plt.plot(t_axis, percent_particles_trapped_or_left_and_axis_bound, '--', color=color, label=label)
    # plt.plot(t_axis, percent_particles_trapped, '--b', label='trapped')
    # plt.plot(t_axis, percent_particles_trapped_and_axis_bound, '--r', label='trapped, and z/l<' + str(z_cutoff))

    t_single_cell = settings['l'] / settings['v_th'] * 1e6
    plt.plot([t_single_cell, t_single_cell], [0, 100], '--', color='grey', label='t_single_cell')

    # plt.xlabel('$t/\\tau_{cyc}$')
    plt.xlabel('$t$ [$\\mu s$]')
    plt.ylabel('%')
    # plt.title('$E_{RF}$=' + str(ERF) + 'kV/m' + ', Rm=' + str(Rm)
    #           + ', $v_{z,res}/v_{th}$=' + str(vz_res)
    #           + ', $\\alpha=\\omega_{RF}/\\omega_{cyc0}=$' + str(alpha)
    #           + ', $v_{RF}/v_{th}$=' + '{:.1f}'.format(v_RF))
    plt.title('stopping metric as a function of time')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    # plt.figure(2)
    # r0 = z[0] + z_offset
    # plt.plot(r0 * np.cos(np.linspace(0, np.pi,100)), r0 * np.sin(np.linspace(0, np.pi,100)), '-k', linewidth=3, alpha=0.3)
    # plt.plot(np.linspace(0, 10,100),np.linspace(0, 10,100) * (np.pi / 2) * np.arcsin(np.sqrt(1/Rm)), '-k', linewidth=3, alpha=0.3)
    # plt.plot(np.linspace(0, -10,100),np.linspace(0, 10,100) * (np.pi / 2) * np.arcsin(np.sqrt(1/Rm)), '-k', linewidth=3, alpha=0.3)
    # plt.xlabel('$z/l$+' + str(z_offset))
    # plt.ylabel('$z/l$+' + str(z_offset))
    # plt.title('$E_{RF}$=' + str(ERF) + 'kV/m' + ', Rm=' + str(Rm)
    #           + ', $v_{z,res}/v_{th}$=' + str(vz_res)
    #           + ', $\\alpha=\\omega_{RF}/\\omega_{cyc0}=$' + str(alpha)
    #           + ', $v_{RF}/v_{th}$=' + '{:.1f}'.format(v_RF))
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()
