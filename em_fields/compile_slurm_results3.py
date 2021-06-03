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

    compiled_file = curr_dir_full + '.pickle'
    if os.path.exists(compiled_file):
        print(curr_dir + ' already compiled, skipping.')
    else:
        print(curr_dir + 'in compilation progess.')

        # extract the total number of sets in this folder
        num_set_files = len(glob.glob(curr_dir + '/set_*'))
        set_files = [curr_dir + '/set_' + str(ind_set) for ind_set in range(num_set_files)]

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

        settings_file = curr_dir_full + '/settings.pickle'
        with open(settings_file, 'rb') as fid:
            data_dict['settings'] = pickle.load(fid)

        field_dict_file = curr_dir_full + '/field_dict.pickle'
        with open(field_dict_file, 'rb') as fid:
            data_dict['field_dict'] = pickle.load(fid)

        data_dict_file = save_dir + '.pickle'
        with open(compiled_file, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
