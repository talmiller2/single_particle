import pickle

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'axes.grid': True})
# plt.rcParams.update({'axes.linewidth': 3})
# plt.rcParams.update({'axes.edgecolor': 'k'})

save_dir_main = '/Users/talmiller/Downloads/single_particle/'
save_dir_main += '/set6/'

set_names = []

v_loop_list = np.round(np.linspace(0.9, 2.5, 10), 2)
alpha_loop_list = np.round(np.linspace(0.5, 2, 10), 2)
# v_loop_list = v_loop_list[0:2]
# alpha_loop_list = alpha_loop_list[0:2]

totol_loop_runs = len(v_loop_list) * len(alpha_loop_list)
rlc_percent_passed = np.nan * np.zeros([len(v_loop_list), len(alpha_loop_list)])

for ind_v, v_loop in enumerate(v_loop_list):
    for ind_alpha, alpha_loop in enumerate(alpha_loop_list):
        run_name = 'tmax_400_B0_0.1_T_3.0_ERF_1_alpha_' + str(alpha_loop) + '_vz_' + str(v_loop)
        save_dir = save_dir_main + run_name

        # load runs data
        data_dict_file = save_dir + '.pickle'
        with open(data_dict_file, 'rb') as fid:
            data_dict = pickle.load(fid)
        settings = data_dict['settings']
        field_dict = data_dict['field_dict']

        t_array = data_dict['t'][0, :]
        num_times = len(t_array)

        inds_rlc = []
        inds_llc = []
        inds_trap = []

        for ind_point in range(data_dict['z'].shape[0]):
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
            elif in_loss_cone and not positive_z_velocity:
                inds_llc += [ind_point]
            else:
                inds_trap += [ind_point]

        # plot what particle % passed some z threshold, as a function of t
        # z_cutoff_list = [5, 10, 20]
        # z_cutoff_list = [10]
        z_cutoff_list = [5]
        # z_cutoff_list = [-2]
        for ind_z_cutoff, z_cutoff in enumerate(z_cutoff_list):
            percent_escaped = np.zeros(len(t_array))
            for k in range(len(t_array)):
                # t_vec += t_array
                # t_vec = np.append(t_vec, t_array / field_dict['tau_cyclotron'])
                inds_curr = inds_rlc
                # inds_curr = inds_trap
                # inds_curr = inds_llc
                percent_escaped[k] = 100 * len(np.where(data_dict['z'][inds_curr, k] / settings['l'] > z_cutoff)[0])
                # percent_escaped[k] = 100 * len(np.where(data_dict['z'][inds_curr, k] / settings['l'] < z_cutoff)[0])
                percent_escaped[k] /= len(inds_curr)

            plt.figure(7 + ind_z_cutoff)
            label = run_name
            plt.plot(t_array / field_dict['tau_cyclotron'], percent_escaped, label=label, linestyle='-')
            plt.xlabel('$t/\\tau_{cyc}$')
            plt.ylabel('% passed')
            plt.title('$z_{cut}/l$=' + str(z_cutoff))
            plt.grid(True)
            # plt.legend()

            rlc_percent_passed[ind_v, ind_alpha] = percent_escaped[-1]

# 2d plot of % passed particles as a function of v, alpha
plt.figure()
sns.heatmap(rlc_percent_passed, xticklabels=v_loop_list, yticklabels=alpha_loop_list)
plt.xlabel('$v/v_{th}$')
plt.ylabel('$\\alpha$')
plt.title('rightLC %passed $z_{cut}/l$=' + str(z_cutoff) + ' at $t/\\tau_{cyc}$=' + str(
    int(t_array[-1] / field_dict['tau_cyclotron'])))
plt.tight_layout()
