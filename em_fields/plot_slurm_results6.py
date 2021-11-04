import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

plt.close('all')

save_dir_main = '/Users/talmiller/Downloads/single_particle/'
# save_dir_main += '/set4/'
# save_dir_main += '/set5/'
# save_dir_main += '/set7_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir_main += '/set8_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir_main += '/set9_T_10keV_B0_1T_Rm_2_l_1_phase_pi/'
# save_dir_main += '/set10_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir_main += '/set11_T_B0_1T_Rm_2_l_1m_randphase/'
# save_dir_main += '/set12_T_B0_1T_Rm_4_l_1m_randphase/'
# save_dir_main += '/set13_T_B0_1T_Rm_2_l_1m_randphase/'
save_dir_main += '/set14_T_B0_1T_Rm_2_l_1m_randphase_save_intervals/'

set_names = []

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

omega_RF_over_omega_cyc_0 = alpha
v_RF = vz_res * alpha / (alpha - 1.0)
print('vz_res/v_th = ' + str(vz_res) + ', alpha = ' + str(alpha))
print('omega_RF/omega_cyc0 = ' + '{:.2f}'.format(omega_RF_over_omega_cyc_0) + ', v_RF/v_th = ' + '{:.2f}'.format(v_RF))

if ERF > 0:
    set_names += ['ERF_' + str(ERF) + '_alpha_' + str(alpha) + '_vz_' + str(vz_res)]
else:
    set_names += ['ERF_0']

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
    # ind_points = range(300)
    ind_points = range(1000)
    # ind_points = range(2000)
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
            percent_particles_trapped = 0 * t
            percent_particles_trapped_and_axis_bound = 0 * t

        # calculate if a particle is initially in right loss cone
        # LC_cutoff = field_dict['Rm'] ** (-0.5)
        # in_loss_cone = v_transverse[0] / v[0] < LC_cutoff
        in_loss_cone = (v_transverse[0] / v[0]) ** 2 < 1 / field_dict['Rm']
        # in_loss_cone = (v_transverse[0] / v_axial[0]) ** 2 < 1 / (field_dict['Rm'] - 1.0)
        positive_z_velocity = v_axial[0] > 0

        if in_loss_cone and positive_z_velocity:  # right loss cone
            linestyle = '-'
            linewidth = 1
            do_plot = True
            # do_plot = False
        elif in_loss_cone and not positive_z_velocity:  # left loss cone
            # linestyle = ':'
            linestyle = '-'
            linewidth = 1
            do_plot = True
            # do_plot = False
        else:  # trapped
            # linestyle = '--'
            linestyle = '-'
            linewidth = 1
            do_plot = True
            # do_plot = False

        # plots
        if do_plot and do_particles_plot:
            # check loss cone criterion as a function of time, for varying Bz(t)
            Rm_dynamic = field_dict['B0'] * field_dict['Rm'] / Bz
            out_of_loss_cone = (v_transverse / v) ** 2 >= 1 / Rm_dynamic

            z_cutoff = 10
            z_axis_bound = z / settings['l'] <= z_cutoff
            trapped = out_of_loss_cone * z_axis_bound

            percent_particles_trapped += 1.0 * out_of_loss_cone
            percent_particles_trapped_and_axis_bound += 1.0 * out_of_loss_cone * z_axis_bound

    percent_particles_trapped /= num_particles * 1.0
    percent_particles_trapped_and_axis_bound /= num_particles * 1.0

    if do_particles_plot:
        plt.figure(77)
        plt.plot(t / field_dict['tau_cyclotron'], percent_particles_trapped, '-b', label='out of LC')
        plt.plot(t / field_dict['tau_cyclotron'], percent_particles_trapped_and_axis_bound, '-r',
                 label='out of LC, and z/l<' + str(z_cutoff))
        plt.xlabel('$t/\\tau_{cyc}$')
        # plt.ylabel('rightLC %passed $z_{cut}/l$=' + str(z_cutoff))
        plt.ylabel('%')
        # plt.title('$z_{cut}/l$=' + str(z_cutoff))
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
