import pickle

import numpy as np

from em_fields.magnetic_forms import get_mirror_magnetic_field
from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

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

# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1_vz_2_sample4pi']
set_names += ['tmax_400_B0_0.1_T_3.0_ERF_1_alpha_1.3_vz_2_sample4pi']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_10_alpha_1_vz_2_sample4pi']
# set_names += ['tmax_400_B0_0.1_T_3.0_ERF_10_alpha_1.3_vz_2_sample4pi']


for set_ind in range(len(set_names)):
    set_name = set_names[set_ind]
    # set_label = set_labels[set_ind]
    set_label = set_name.split('T_3.0_')[-1]

    save_dir = save_dir_main + set_name
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
    num_particles = len(data_dict['z'])
    # ind_points = [0, 1, 2, 4, 5]
    # ind_points = [0]
    # ind_points = [4]
    # ind_points = range(5)
    # ind_points = range(10)
    # ind_points = range(30)
    # ind_points = range(100)
    # ind_points = range(100, 200)
    # ind_points = range(20, 30)
    # ind_points = range(30, 40)
    ind_points = range(num_particles)

    do_particles_plot = False
    # do_particles_plot = True

    if do_particles_plot:
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
        # fig = plt.figure(1, figsize=(16, 6))
        fig = plt.figure(1)
        if fig.axes == []:
            ax1 = plt.subplot(1, 4, 1)
            ax2 = plt.subplot(1, 4, 2)
            ax3 = plt.subplot(1, 4, 3)
            ax4 = plt.subplot(1, 4, 4)
        else:
            [ax1, ax2, ax3, ax4] = fig.axes

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
        LC_cutoff = field_dict['Rm'] ** (-0.5)
        in_loss_cone = v_transverse[0] / v[0] < LC_cutoff
        positive_z_velocity = v_axial[0] > 0

        if in_loss_cone and positive_z_velocity:
            inds_rlc += [ind_point]
            linestyle = '-'
            linewidth = 1
            # do_plot = True
            do_plot = False
        elif in_loss_cone and not positive_z_velocity:
            inds_llc += [ind_point]
            linestyle = ':'
            linewidth = 2
            do_plot = True
            # do_plot = False
        else:
            inds_trap += [ind_point]
            linestyle = '--'
            linewidth = 1
            # do_plot = True
            do_plot = False


        # record the fractions of particles in each population as a function of time
        if in_loss_cone and positive_z_velocity:
            inds = np.where(v_transverse / v < LC_cutoff)[0]
            # inds_times_in_loss_cone = [i for i in range(num_times) if v_transverse[i] / v[i] <= LC_cutoff and z[i] >= z[0]]
            cnt_rlc_array[inds] += 1
        elif in_loss_cone and not positive_z_velocity:
            inds = np.where(v_transverse / v < LC_cutoff)[0]
            cnt_llc_array[inds] += 1
        else:
            inds = np.where(v_transverse / v >= LC_cutoff)[0]
            cnt_trap_array[inds] += 1

        # plots
        if do_plot and do_particles_plot:
            ax1.plot(t_array / field_dict['tau_cyclotron'], z / settings['l'], label=ind_point, linestyle=linestyle,
                     linewidth=linewidth)
            ax1.set_xlabel('$t/\\tau_{cyc}$')
            ax1.set_ylabel('$z/l$')

            ax2.plot(t_array / field_dict['tau_cyclotron'], v / settings['v_th'], label=ind_point, linestyle=linestyle,
                     linewidth=linewidth)
            ax2.set_xlabel('$t/\\tau_{cyc}$')
            ax2.set_ylabel('$|v|/v_{th}$')

            ax3.plot(t_array / field_dict['tau_cyclotron'], v_transverse / settings['v_th'], label=ind_point,
                     linestyle=linestyle, linewidth=linewidth)
            ax3.set_xlabel('$t/\\tau_{cyc}$')
            ax3.set_ylabel('$v_{\perp}/v_{th}$')

            # ax4.plot(v_axial / settings['v_th'], label=ind_point, linestyle=linestyle, linewidth=linewidth)
            ax4.plot(t_array / field_dict['tau_cyclotron'], v_transverse / v, label=ind_point, linestyle=linestyle,
                     linewidth=linewidth)
            ax4.set_xlabel('$t/\\tau_{cyc}$')
            # ax4.set_ylabel('$v_{z}/v_{th}$')
            ax4.set_ylabel('$v_{\perp}/|v|$')

    if do_particles_plot:
        ax4.plot(t_array / field_dict['tau_cyclotron'], LC_cutoff * np.ones(len(t_array)), '-k', label='LC cutoff',
                 linewidth=3)
        ax4.legend()
        fig.tight_layout()

    # plt.figure(2)
    # plt.plot(t_array / field_dict['tau_cyclotron'], cnt_rlc_array / cnt_rlc_array[0], label='right LC')
    # plt.plot(t_array / field_dict['tau_cyclotron'], cnt_llc_array / cnt_llc_array[0], label='left LC')
    # plt.plot(t_array / field_dict['tau_cyclotron'], cnt_trap_array / cnt_trap_array[0], label='trapped')
    # plt.xlabel('$t/\\tau_{cyc}$')
    # plt.legend()

    ## average z plot of each population
    # z_avg_rlc = 0 * t_array
    # z_avg_llc = 0 * t_array
    # z_avg_trap = 0 * t_array
    # for ind_t in range(num_times):
    #     z_avg_rlc[ind_t] = np.mean(data_dict['z'][inds_rlc, ind_t])
    #     z_avg_llc[ind_t] = np.mean(data_dict['z'][inds_llc, ind_t])
    #     z_avg_trap[ind_t] = np.mean(data_dict['z'][inds_trap, ind_t])

    # plt.figure(3)
    # plt.plot(t_array / field_dict['tau_cyclotron'], z_avg_rlc /  settings['l'], label='right LC')
    # # plt.plot(t_array / field_dict['tau_cyclotron'], z_avg_llc /  settings['l'], label='left LC')
    # # plt.plot(t_array / field_dict['tau_cyclotron'], z_avg_trap /  settings['l'], label='trapped LC')
    # plt.xlabel('$t/\\tau_{cyc}$')
    # plt.ylabel('$z/l$')
    # plt.legend()

    ## histogram of z values in several different times
    # plt.figure(4, figsize=(6,3))
    # plt.figure(4, figsize=(6,6))
    # # plt.subplot(2,1,1)
    # plt.subplot(len(set_names), 1, set_ind+1)
    #
    # color = None
    # # color = 'b'
    # # color = 'r'
    # # fracs = [0.2, 0.5, 0.8, 1.0]
    # fracs = [0.1, 0.5, 1.0]
    # # fracs = [1.0]
    # for frac in fracs:
    #     ind_t = int(num_times * frac) - 1
    #     t_norm_str = str(int(t_array[ind_t] / field_dict['tau_cyclotron']))
    #     alpha = 0.5
    #     bins = np.linspace(0, 40, 40)
    #     label = set_label + ', $t/\\tau_{cyc}$=' + t_norm_str
    #     plt.hist(data_dict['z'][inds_rlc, ind_t] / settings['l'], label=label, density=True, alpha=alpha, bins=bins, color=color)
    #     # plt.hist(data_dict['z'][inds_llc, ind_t] / settings['l'], label='lLC $t/\\tau_{cyc}$=' + t_norm_str, density=True, alpha=alpha, bins=bins, color=color)
    #     # plt.hist(data_dict['z'][inds_trap, ind_t] / settings['l'], label='trapped $t/\\tau_{cyc}$=' + t_norm_str, density=True, alpha=alpha, bins=bins, color=color)
    # plt.gca().set_yticks([])
    # if set_ind  == len(set_names) - 1:
    #     plt.xlabel('$z/l$')
    #     plt.tight_layout()
    # plt.legend()

    # plot 2d histogram of particles in different z, t
    t_vec = []
    z_vec = []
    for k in range(len(t_array)):
        # t_vec += t_array
        # t_vec = np.append(t_vec, t_array / field_dict['tau_cyclotron'])
        # inds_curr = inds_rlc
        # inds_curr = inds_trap
        # inds_curr = inds_llc
        inds_curr = inds_rlc + inds_trap + inds_llc
        t_vec = np.append(t_vec, t_array[k] / field_dict['tau_cyclotron'] * np.ones(len(inds_curr)))
        z_vec = np.append(z_vec, data_dict['z'][inds_curr, k] / settings['l'])
    # plt.figure(5, figsize=(10, 5))
    # plt.subplot(1, len(set_names), set_ind + 1)
    # # bins = [1 + t_array / field_dict['tau_cyclotron'], np.linspace(0, 40, 40)]
    # # bins = [1 + t_array / field_dict['tau_cyclotron'], np.linspace(-1, 3, 40)]
    # # bins = [1 + t_array / field_dict['tau_cyclotron'], np.linspace(-10, 10, 20)]
    # # bins = [1 + t_array / field_dict['tau_cyclotron'], np.linspace(-20, 2, 40)]
    # # bins = [1 + t_array / field_dict['tau_cyclotron'], np.linspace(0, 20, 40)]
    # bins = [1 + t_array / field_dict['tau_cyclotron'], np.linspace(2, 20, 40)]
    # # bins = 50
    # plt.hist2d(t_vec, z_vec, bins=bins)
    # if set_ind == 0:
    #     plt.ylabel('$z/l$')
    # plt.xlabel('$t/\\tau_{cyc}$')
    # # plt.title(set_label)
    # plt.title(set_ind)
    # plt.tight_layout()

    # plot 2d histogram of particles in different v, t
    t_vec = []
    v_vec = []
    for k in range(len(t_array)):
        # t_vec += t_array
        # t_vec = np.append(t_vec, t_array / field_dict['tau_cyclotron'])
        inds_curr = inds_rlc
        # inds_curr = inds_trap
        # inds_curr = inds_llc
        t_vec = np.append(t_vec, t_array[k] / field_dict['tau_cyclotron'] * np.ones(len(inds_curr)))
        v_vec = np.append(v_vec, data_dict['v'][inds_curr, k] / settings['v_th'])
    # plt.figure(6, figsize=(10,5))
    # plt.subplot(1, len(set_names), set_ind+1)
    # bins = [1 + t_array / field_dict['tau_cyclotron'], np.linspace(0, 20, 40)]
    # # bins = 50
    # plt.hist2d(t_vec, v_vec, bins=bins)
    # if set_ind == 0:
    #     plt.ylabel('$v/v_{th}$')
    # plt.xlabel('$t/\\tau_{cyc}$')
    # plt.title(set_ind)
    # plt.tight_layout()

    ## Up to time T, how many particles were stopped (z direction flipped), and what is the average z it happend in?
    # plot what particle % passed some z threshold, as a function of t
    # z_cutoff_list = [2, 4 ,6]
    # z_cutoff_list = [5, 10, 20]
    # z_cutoff_list = [10]
    z_cutoff_list = [5]
    # z_cutoff_list = [-5]
    for ind_z_cutoff, z_cutoff in enumerate(z_cutoff_list):
        percent_escaped = np.zeros(len(cnt_rlc_array))
        for k in range(len(t_array)):
            # t_vec += t_array
            # t_vec = np.append(t_vec, t_array / field_dict['tau_cyclotron'])
            # inds_curr = inds_rlc
            # inds_curr = inds_trap
            # inds_curr = inds_llc
            inds_curr = inds_rlc + inds_trap + inds_llc
            percent_escaped[k] = 100 * len(np.where(data_dict['z'][inds_curr, k] / settings['l'] > z_cutoff)[0])
            # percent_escaped[k] = 100 * len(np.where(data_dict['z'][inds_curr, k] / settings['l'] < z_cutoff)[0])
            percent_escaped[k] /= len(inds_curr)

        # plt.figure(7 + ind_z_cutoff)
        plt.figure(7)
        # label = set_ind
        # label = set_label
        label = set_label + ', $z_{cut}/l$=' + str(z_cutoff)
        # label = '$z_{cut}/l$=' + str(z_cutoff)
        if set_ind == 0:
            linestyle = '-'
        else:
            linestyle = '--'
        plt.plot(t_array / field_dict['tau_cyclotron'], percent_escaped, label=label, linestyle=linestyle)
        plt.xlabel('$t/\\tau_{cyc}$')
        plt.ylabel('% passed')
        # plt.title('$z_{cut}/l$=' + str(z_cutoff))
        plt.grid(True)
        plt.legend()

    # calculate the magnetic field during evolution to check the loss cone status
    loss_cone_condition_mat = np.zeros([len(ind_points), len(t_array)])

    for ind_point in ind_points:

        t = t_array
        z = data_dict['z'][ind_point, :]
        v = data_dict['v'][ind_point, :]
        v_r = data_dict['v_transverse'][ind_point, :]
        B0 = field_dict['B0']
        Rm = field_dict['Rm']
        B_max = B0 * Rm
        l = field_dict['l']
        mirror_field_type = field_dict['mirror_field_type']
        Bz = []
        loss_cone_condition = []
        for iz, z_curr in enumerate(z):
            x_curr = np.array([0, 0, z_curr])
            B = get_mirror_magnetic_field(x_curr, B0, Rm, l, use_transverse_fields=True, z0=0,
                                          mirror_field_type=mirror_field_type)
            Bz_curr = B[2]
            Bz += [Bz_curr]
            # Rm_curr = Bz_curr / B_max
            Rm_curr = B_max / Bz_curr
            loss_cone_condition += [(v_r[iz] / v[iz]) ** 2.0 - (Rm_curr) ** (-1.0)]

        loss_cone_condition_mat[ind_point, :] = np.array(loss_cone_condition)

    percent_stay_right = np.zeros(len(t_array))
    for k in range(len(t_array)):
        percent_stay_right[k] = len(np.where(loss_cone_condition_mat[inds_rlc, k] < 0)[0])
    percent_stay_right *= 100.0 / len(inds_rlc)

    percent_stay_left = np.zeros(len(t_array))
    for k in range(len(t_array)):
        percent_stay_left[k] = len(np.where(loss_cone_condition_mat[inds_llc, k] < 0)[0])
    percent_stay_left *= 100.0 / len(inds_llc)

    percent_stay_trapped = np.zeros(len(t_array))
    for k in range(len(t_array)):
        percent_stay_trapped[k] = len(np.where(loss_cone_condition_mat[inds_trap, k] >= 0)[0])
    percent_stay_trapped *= 100.0 / len(inds_trap)

    # fit exponents
    # p = np.polyfit(t_array, np.log(percent_stay_right), 1, w=np.sqrt(percent_stay_right))
    p = np.polyfit(t_array[0:3], percent_stay_right[0:3], 1)
    lin_fit = np.polyval(p, t_array)
    lin_fit[np.where(lin_fit <= 0)] = np.nan

    plt.figure(8)
    if set_ind == 0:
        color = 'b'
    else:
        color = 'r'
    set_label = ''
    label = set_label + ' right'
    plt.plot(t_array / field_dict['tau_cyclotron'], percent_stay_right, label=label, color=color, linestyle='-')
    # plt.plot(t_array / field_dict['tau_cyclotron'], np.exp(np.polyval(p, t_array)), label=label, color='k', linestyle='-')
    # plt.plot(t_array / field_dict['tau_cyclotron'], np.polyval(p, t_array), label=label, color='k', linestyle='-')
    plt.plot(t_array / field_dict['tau_cyclotron'], lin_fit, label=label + ' slope', color='k', linestyle='--')

    label = set_label + ' left'
    plt.plot(t_array / field_dict['tau_cyclotron'], percent_stay_left, label=label, color=color, linestyle='--')
    label = set_label + ' trapped'
    plt.plot(t_array / field_dict['tau_cyclotron'], percent_stay_trapped, label=label, color=color, linestyle=':')
    plt.xlabel('$t/\\tau_{cyc}$')
    plt.ylabel('% that stay in original population')
    plt.title('using loss cone criteria')
    plt.grid(True)
    plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    plt.tight_layout()
