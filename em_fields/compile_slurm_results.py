import os

import numpy as np
from scipy.io import savemat

save_dir = '/home/talm/code/single_particle/slurm_runs/'
# save_dir += '/set1/'
# save_dir += '/set2/'
save_dir += '/set3/'

os.chdir(save_dir)

# v_abs_list = np.linspace(0.5, 1.5, 21)
# angle_to_z_axis_list = [i for i in range(0, 181, 5)]
# phase_RF_list = np.array([0, 0.25, 0.5]) * np.pi

v_abs_list = np.linspace(0.7, 1.3, 21)
angle_to_z_axis_list = [i for i in range(0, 91, 5)]
phase_RF_list = np.array([0]) * np.pi

total_number_of_combinations = 1
total_number_of_combinations *= len(v_abs_list)
total_number_of_combinations *= len(angle_to_z_axis_list)
total_number_of_combinations *= len(phase_RF_list)

# run over dirs and compile the runs where needed
run_dirs = [curr_dir for curr_dir in os.listdir(save_dir) if os.path.isdir(curr_dir)]
for curr_dir in run_dirs:
    curr_dir_full = save_dir + '/' + curr_dir
    complied_mat_file = curr_dir_full + '.mat'

    if not os.path.exists(complied_mat_file):
        print(curr_dir)
        compiled_mat_dict = {}
        compiled_mat_dict['z'] = np.nan * np.zeros(total_number_of_combinations)
        compiled_mat_dict['E'] = np.nan * np.zeros(total_number_of_combinations)
        compiled_mat_dict['v_r_mean'] = np.nan * np.zeros(total_number_of_combinations)
        compiled_mat_dict['v_z_mean'] = np.nan * np.zeros(total_number_of_combinations)
        cnt = 0
        for v_abs in v_abs_list:
            for angle_to_z_axis in angle_to_z_axis_list:
                for phase_RF in phase_RF_list:
                    run_name = ''
                    run_name += 'v_' + '{:.2f}'.format(v_abs)
                    run_name += '_angle_' + str(angle_to_z_axis)
                    run_name += '_phaseRF_' + '{:.2f}'.format(phase_RF / np.pi)
                    try:
                        data = np.loadtxt(curr_dir_full + '/' + run_name + '.txt')
                    except:
                        print('failed to load ' + run_name + ', setting NaNs instead.')
                        data = [np.nan, np.nan, np.nan, np.nan]
                    compiled_mat_dict['z'][cnt] = data[0]
                    compiled_mat_dict['E'][cnt] = data[1]
                    compiled_mat_dict['v_r_mean'][cnt] = data[2]
                    compiled_mat_dict['v_z_mean'][cnt] = data[3]
                    cnt += 1

        savemat(complied_mat_file, compiled_mat_dict)
