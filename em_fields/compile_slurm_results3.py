import glob
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
    if os.path.exists(compiled_mat_file):
        print(curr_dir + ' already compiled, skipping.')
    else:
        print(curr_dir + 'in compilation progess.')

        # extract the total number of sets in this folder
        set_files = glob.glob(curr_dir + '/set*')

        # define the mat_dict where all data will be compiled
        run_info_file = curr_dir_full + '/run_info.mat'
        mat_dict = loadmat(run_info_file)

        # loop over all saved sets and combine their data
        for ind_set, set_file in enumerate(set_files):
            print('set # ' + str(ind_set))
            set_mat_dict = loadmat(set_file)
            keys = [key for key in set_mat_dict.keys() if '__' not in key]
            for key in keys:
                if ind_set == 0:
                    mat_dict[key] = set_mat_dict[key]
                else:
                    mat_dict[key] = np.vstack([mat_dict[key], set_mat_dict[key]])
        savemat(compiled_mat_file, mat_dict)
