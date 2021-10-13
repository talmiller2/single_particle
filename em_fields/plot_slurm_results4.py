import pickle

import numpy as np

from em_fields.magnetic_forms import get_mirror_magnetic_field
from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# plt.close('all')

save_dir_main = '/Users/talmiller/Downloads/single_particle/'
# save_dir_main += '/set4/'
save_dir_main += '/set5/'

set_names = []

# set_names += ['tmax_400_B0_0.1_T_3.0_traveling_ERF_0_alpha_2.718']
# set_names += ['tmax_400_B0_0.1_T_3.0_traveling_ERF_2_alpha_2.718']


# set_names += ['tmax_601_B0_0.1_T_3.0_ERF_0_alpha_1.1_vz_2']
# set_names += ['tmax_601_B0_0.1_T_3.0_ERF_5_alpha_1.1_1.1_1.1_vz_1_1.5_2']
# set_names += ['tmax_601_B0_0.1_T_3.0_ERF_5_alpha_1.1_1.1_vz_1_2']
# set_names += ['tmax_601_B0_0.1_T_3.0_ERF_5_alpha_1.1_vz_1']
# set_names += ['tmax_601_B0_0.1_T_3.0_ERF_5_alpha_1.1_vz_2']

# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_0_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.1_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.1_1.1_vz_1_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.1_1.1_1.1_vz_1_1.5_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_3_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_3_alpha_1.1_vz_2']

# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.5_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.7_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.9_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.3_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.7_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_2_vz_1']

# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.9_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.95_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.96_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.97_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_0.98_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.3_vz_2']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.7_vz_2']

# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_0_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_3_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_5_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_10_alpha_1.1_vz_1']


# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_5_alpha_1.1_vz_1']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_5_alpha_1.1_vz_1_zeroBRF']

set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1_vz_2_sample4pi']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.3_vz_2_sample4pi']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_10_alpha_1_vz_2_sample4pi']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_10_alpha_1.3_vz_2_sample4pi']


for set_ind in range(len(set_names)):
    set_name = set_names[set_ind]
    # set_label = set_labels[set_ind]
    set_label = set_name.split('T_3.0_')[-1]

    save_dir = save_dir_main + set_name

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
    num_particles = len(data_dict['z'])
    # ind_points = [0, 1, 2, 4, 5]
    # ind_points = [0]
    # ind_points = [3]
    # ind_points = range(5)
    # ind_points = range(10)
    # ind_points = range(20)
    ind_points = range(100)
    # ind_points = range(300)
    # ind_points = range(1000)
    # ind_points = range(100, 200)
    # ind_points = range(20, 30)
    # ind_points = range(30, 40)
    # ind_points = range(num_particles)

    # do_particles_plot = False
    do_particles_plot = True

    if do_particles_plot:
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
        # fig = plt.figure(1, figsize=(16, 6))
        # # fig = plt.figure(1)
        # if fig.axes == []:
        #     ax1 = plt.subplot(1, 4, 1)
        #     ax2 = plt.subplot(1, 4, 2)
        #     ax3 = plt.subplot(1, 4, 3)
        #     ax4 = plt.subplot(1, 4, 4)
        # else:
        #     [ax1, ax2, ax3, ax4] = fig.axes

        # fig = plt.figure(2, figsize=(12, 5))
        # if fig.axes == []:
        #     ax1 = plt.subplot(1, 2, 1)
        #     ax2 = plt.subplot(1, 2, 2)
        # else:
        #     [ax1, ax2] = fig.axes

        pass

    t_array = data_dict['t'][0, :]
    num_times = len(t_array)

    inds_rlc = []
    inds_llc = []
    inds_trap = []

    cnt_rlc_array = 0 * t_array
    cnt_llc_array = 0 * t_array
    cnt_trap_array = 0 * t_array

    for ind_point in ind_points:
        z = data_dict['z'][ind_point, :]
        v = data_dict['v'][ind_point, :]
        v_transverse = data_dict['v_transverse'][ind_point, :]
        v_axial = data_dict['v_axial'][ind_point, :]

        # calculate if a particle is initially in right loss cone
        # LC_cutoff = field_dict['Rm'] ** (-0.5)
        # in_loss_cone = v_transverse[0] / v[0] < LC_cutoff
        in_loss_cone = (v_transverse[0] / v[0]) ** 2 < 1 / field_dict['Rm']
        # in_loss_cone = (v_transverse[0] / v_axial[0]) ** 2 < 1 / (field_dict['Rm'] - 1.0)
        positive_z_velocity = v_axial[0] > 0

        if in_loss_cone and positive_z_velocity:  # right loss cone
            inds_rlc += [ind_point]
            linestyle = '-'
            linewidth = 1
            do_plot = True
            # do_plot = False
        elif in_loss_cone and not positive_z_velocity:  # left loss cone
            inds_llc += [ind_point]
            # linestyle = ':'
            linestyle = '-'
            linewidth = 1
            do_plot = True
            # do_plot = False
        else:  # trapped
            inds_trap += [ind_point]
            # linestyle = '--'
            linestyle = '-'
            linewidth = 1
            do_plot = True
            # do_plot = False

        # TODO: assign B for each point in trajectory and use only points that cross the minimal B during evolution
        t_array = data_dict['t'][0, :]
        B0 = field_dict['B0']
        Rm = field_dict['Rm']
        B_max = B0 * Rm
        l = field_dict['l']
        mirror_field_type = field_dict['mirror_field_type']
        Bz = []
        for iz, z_curr in enumerate(z):
            x_curr = np.array([0, 0, z_curr])
            B = get_mirror_magnetic_field(x_curr, B0, Rm, l, mirror_field_type=mirror_field_type)
            Bz_curr = B[2]
            Bz += [Bz_curr]

        # pick only times where particle is close to cell center / minimal mirror magnetic field
        Bz = np.array(Bz)
        # inds_cell_center = np.where(abs((Bz- B0)/B0) < 0.01)
        # t_array = t_array[inds_cell_center]
        # z = z[inds_cell_center]
        # v = v[inds_cell_center]
        # # v_axial = v_axial[inds_cell_center]
        # v_axial = abs(v_axial[inds_cell_center])
        # v_transverse = v_transverse[inds_cell_center]
        # Bz = Bz[inds_cell_center]

        # transform each point on the trajectory to what it would be at the magnetic field minimum
        v_transverse_tag = v_transverse * np.sqrt(B0 / Bz)
        v_axial_tag = np.sqrt(v_axial ** 2 + v_transverse ** 2 * (1 - B0 / Bz))
        v_transverse = v_transverse_tag
        v_axial = v_axial_tag

        if not positive_z_velocity:  # draw points that started with negative velocity, on negative side of the plot
            v_axial *= -1

        # plots
        if do_plot and do_particles_plot:
            # ax1.plot(t_array / field_dict['tau_cyclotron'], z / settings['l'], label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth)
            # ax1.set_xlabel('$t/\\tau_{cyc}$')
            # ax1.set_ylabel('$z/l$')
            #
            # ax2.plot(t_array / field_dict['tau_cyclotron'], v / settings['v_th'], label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth)
            # ax2.set_xlabel('$t/\\tau_{cyc}$')
            # ax2.set_ylabel('$|v|/v_{th}$')
            #
            # ax3.plot(t_array / field_dict['tau_cyclotron'], v_transverse / settings['v_th'], label=ind_point,
            #          linestyle=linestyle, linewidth=linewidth)
            # ax3.set_xlabel('$t/\\tau_{cyc}$')
            # ax3.set_ylabel('$v_{\perp}/v_{th}$')
            #
            # # ax4.plot(v_axial / settings['v_th'], label=ind_point, linestyle=linestyle, linewidth=linewidth)
            # ax4.plot(t_array / field_dict['tau_cyclotron'], v_transverse / v, label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth)
            # ax4.set_xlabel('$t/\\tau_{cyc}$')
            # # ax4.set_ylabel('$v_{z}/v_{th}$')
            # ax4.set_ylabel('$v_{\perp}/|v|$')

            # plt.figure(2)
            # E = (v / settings['v_th']) ** 2
            # E_transverse = (v_transverse / settings['v_th']) ** 2
            # plt.plot(t_array / field_dict['tau_cyclotron'], E_transverse, label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth)
            #
            # plt.figure(3)
            # plt.plot(t_array / field_dict['tau_cyclotron'], z / settings['l'], label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth, marker='o', markersize=2,)

            plt.figure(4)
            plt.plot(v_axial / settings['v_th'], v_transverse / settings['v_th'], label=ind_point, linewidth=linewidth,
                     # linestyle=linestyle,
                     linestyle='none', marker='o', markersize=2,
                     )
            plt.plot(v_axial[0] / settings['v_th'], v_transverse[0] / settings['v_th'], 'ko', markersize=2)

            # plt.figure(5)
            # plt.plot(t_array / field_dict['tau_cyclotron'], Bz, label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth)

            # E = (v / settings['v_th']) ** 2
            # E_transverse = (v_transverse / settings['v_th']) ** 2
            # ax1.plot(t_array / field_dict['tau_cyclotron'], E_transverse, label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth)
            #
            # ax2.plot(t_array / field_dict['tau_cyclotron'], z / settings['l'], label=ind_point, linestyle=linestyle,
            #          linewidth=linewidth)

    if do_particles_plot:
        # ax1.grid(True)
        # ax2.grid(True)
        # ax3.grid(True)
        # ax4.grid(True)
        #
        # ax4.plot(t_array / field_dict['tau_cyclotron'], LC_cutoff * np.ones(len(t_array)), '-k', label='LC cutoff',
        #          linewidth=3)
        # # ax4.legend()

        # plt.figure(2)
        # plt.grid(True)
        # plt.xlabel('$t/\\tau_{cyc}$')
        # plt.ylabel('$E_{\perp}/E_{th}$')
        # plt.tight_layout()
        #
        # plt.figure(3)
        # plt.grid(True)
        # plt.xlabel('$t/\\tau_{cyc}$')
        # plt.ylabel('$z/l$')
        # plt.tight_layout()

        plt.figure(4)
        plt.plot(np.linspace(0, 2, 10), np.sqrt(1 / (field_dict['Rm'] - 1.0)) * np.linspace(0, 2, 10), '-k',
                 linewidth=3)
        plt.plot(-np.linspace(0, 2, 10), np.sqrt(1 / (field_dict['Rm'] - 1.0)) * np.linspace(0, 2, 10), '-k',
                 linewidth=3)
        plt.grid(True)
        plt.xlabel('$v_{\parallel}/v_{th}$')
        plt.ylabel('$v_{\perp}/v_{th}$')
        plt.tight_layout()

        # plt.figure(5)
        # plt.grid(True)
        # plt.xlabel('$t/\\tau_{cyc}$')
        # plt.ylabel('$B_z$')
        # plt.tight_layout()

        # ax1.grid(True)
        # ax1.set_xlabel('$t/\\tau_{cyc}$')
        # ax1.set_ylabel('$E_{\perp}/E_{th}$')
        # ax2.grid(True)
        # ax2.set_xlabel('$t/\\tau_{cyc}$')
        # ax2.set_ylabel('$z/l$')
        # fig.tight_layout()
