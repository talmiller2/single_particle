import os

import numpy as np
from scipy.io import savemat

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set1/'

v_abs_list = np.linspace(0.5, 1.5, 21)
angle_to_z_axis_list = [i for i in range(0, 181, 5)]
phase_RF_list = np.array([0, 0.25, 0.5]) * np.pi

total_number_of_combinations = 1
total_number_of_combinations *= len(v_abs_list)
total_number_of_combinations *= len(angle_to_z_axis_list)
total_number_of_combinations *= len(phase_RF_list)

# run over dirs and compile the runs where needed
dirs = os.listdir(save_dir)
for dir in dirs:
    print(dir)
    curr_dir = save_dir + '/' + dir
    complied_mat_file = curr_dir + '.mat'

    if not os.path.exists(complied_mat_file):
        compiled_mat_dict = {}
        compiled_mat_dict['z'] = np.nan * np.zeros(total_number_of_combinations)
        compiled_mat_dict['E'] = np.nan * np.zeros(total_number_of_combinations)
        cnt = 0
        for v_abs in v_abs_list:
            for angle_to_z_axis in angle_to_z_axis_list:
                for phase_RF in phase_RF_list:
                    run_name = ''
                    run_name += 'v_' + '{:.2f}'.format(v_abs)
                    run_name += '_angle_' + str(angle_to_z_axis)
                    run_name += '_phaseRF_' + '{:.2f}'.format(phase_RF / np.pi)
                    data = np.loadtxt(curr_dir + '/' + run_name + '.txt')
                    compiled_mat_dict['z'][cnt] = data[0]
                    compiled_mat_dict['E'][cnt] = data[1]
                    cnt += 1

        savemat(complied_mat_file, compiled_mat_dict)
