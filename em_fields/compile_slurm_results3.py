import glob
import os
import pickle

import numpy as np
from scipy.io import loadmat

save_dir = '/home/talm/code/single_particle/slurm_runs/'
save_dir += '/set4/'

os.chdir(save_dir)

# run over dirs and compile the runs where needed
run_dirs = [curr_dir for curr_dir in os.listdir(save_dir) if os.path.isdir(curr_dir)]
for curr_dir in run_dirs:
    curr_dir_full = save_dir + '/' + curr_dir

    # compiled_mat_file = curr_dir_full + '.mat'
    # if os.path.exists(compiled_mat_file):
    compiled_file = curr_dir_full + '.pickle'
    if os.path.exists(compiled_file):
        print(curr_dir + ' already compiled, skipping.')
    else:
        print(curr_dir + 'in compilation progess.')

        # extract the total number of sets in this folder
        set_files = glob.glob(curr_dir + '/set*')

        # define the mat_dict where all data will be compiled
        run_info_file = curr_dir_full + '/runs_dict.mat'
        data_dict = loadmat(run_info_file)

        # loop over all saved sets and combine their data
        for ind_set, set_file in enumerate(set_files):
            print('set # ' + str(ind_set))
            set_mat_dict = loadmat(set_file)
            keys = [key for key in set_mat_dict.keys() if '__' not in key]
            for key in keys:
                if ind_set == 0:
                    data_dict[key] = set_mat_dict[key]
                else:
                    data_dict[key] = np.vstack([data_dict[key], set_mat_dict[key]])
        # savemat(compiled_mat_file, mat_dict)

        settings_file = curr_dir_full + '/settings.mat'
        data_dict['settings'] = loadmat(settings_file)

        field_dict_file = curr_dir_full + '/field_dict.mat'
        data_dict['field_dict'] = loadmat(field_dict_file)

        with open(compiled_file, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
