import os

import numpy as np
from scipy.io import savemat, loadmat

save_dir = '/home/talm/code/single_particle/slurm_runs/'
save_dir += '/set4/'

os.chdir(save_dir)

# run over dirs and compile the runs where needed
run_dirs = [curr_dir for curr_dir in os.listdir(save_dir) if os.path.isdir(curr_dir)]
for curr_dir in run_dirs:
    curr_dir_full = save_dir + '/' + curr_dir

    compiled_mat_file = curr_dir_full + '.mat'
    if not os.path.exists(compiled_mat_file):
        print(curr_dir)

        # extract the total number of points in this folder
        points_file = curr_dir_full + '/points.mat'
        mat_dict = loadmat(points_file)
        total_number_of_combinations = len(mat_dict['v_0'])

        # define the mat_dict where all data will be compiled
        mat_dict['z'] = []
        mat_dict['E'] = []
        mat_dict['E_transverse'] = []

        # loop over all saved runs and collect their data
        for ind_point in range(total_number_of_combinations):
            run_name = 'ind_' + str(ind_point)
            try:
                data = np.loadtxt(curr_dir_full + '/' + run_name + '.txt')
                mat_dict['z'] += [data[0, :]]
                mat_dict['E'] += [data[1, :]]
                mat_dict['E_transverse'] += [data[2, :]]
            except:
                print('failed to load ' + run_name + ', skipping to next one.')

        savemat(compiled_mat_file, mat_dict)
