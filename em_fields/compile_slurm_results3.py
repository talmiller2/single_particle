import glob
import os
import pickle

import numpy as np
from scipy.io import loadmat

save_dir = '/home/talm/code/single_particle/slurm_runs/'
# save_dir += '/set5/'
# save_dir += '/set6/'
# save_dir += '/set7_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir += '/set8_T_10keV_B0_1T_Rm_2_l_1m/'
# save_dir += '/set9_T_10keV_B0_1T_Rm_2_l_1_phase_pi/'
# save_dir += '/set11_T_B0_1T_Rm_2_l_1m_randphase/'
# save_dir += '/set12_T_B0_1T_Rm_4_l_1m_randphase/'
# save_dir += '/set13_T_B0_1T_Rm_2_l_1m_randphase/'
# save_dir += '/set14_T_B0_1T_Rm_2_l_1m_randphase_save_intervals/'
# save_dir += '/set15_T_B0_1T_l_1m_Logan_intervals/'
# save_dir += '/set16_T_B0_1T_l_1m_Post_intervals/'
# save_dir += '/set17_T_B0_1T_l_3m_Post_intervals/'
# save_dir += '/set18_T_B0_1T_l_3m_Logan_intervals/'
# save_dir += '/set19_T_B0_1T_l_3m_Post_intervals_Rm_1.3/'
# save_dir += '/set20_B0_1T_l_3m_Post_intervals_Rm_3/'
# save_dir += '/set21_B0_1T_l_3m_Post_intervals_Rm_3_different_phases/'
save_dir += '/set22_B0_1T_l_3m_Post_intervals_Rm_3/'

os.chdir(save_dir)

# run over dirs and compile the runs where needed
run_dirs = [curr_dir for curr_dir in os.listdir(save_dir) if os.path.isdir(curr_dir)]
for curr_dir in run_dirs:
    curr_dir_full = save_dir + '/' + curr_dir

    settings_file = curr_dir_full + '/settings.pickle'
    with open(settings_file, 'rb') as fid:
        settings = pickle.load(fid)

    field_dict_file = curr_dir_full + '/field_dict.pickle'
    with open(field_dict_file, 'rb') as fid:
        field_dict = pickle.load(fid)

    print('###############')
    compiled_file = curr_dir_full + '.pickle'
    if os.path.exists(compiled_file):
        print(curr_dir + ' already compiled, skipping.')
    else:
        print(curr_dir + ' in compilation progress.')

        # extract the total number of sets in this folder
        num_set_files = len(glob.glob(curr_dir + '/set_*'))
        set_files = [curr_dir + '/set_' + str(ind_set) for ind_set in range(num_set_files)]

        # define the dict where all data will be compiled
        points_dict_file = curr_dir_full + '/points_dict.mat'
        data_dict = loadmat(points_dict_file)

        # loop over all saved sets and combine their data
        for ind_set, set_file in enumerate(set_files):
            print('   set # ' + str(ind_set))

            if settings['set_save_format'] == 'mat':
                # in this format all data was saved as 2d matrices
                set_dict = loadmat(set_file + '.mat')
                keys = [key for key in set_dict.keys() if '__' not in key]
                for key in keys:
                    if ind_set == 0:
                        data_dict[key] = set_dict[key]
                    else:
                        data_dict[key] = np.vstack([data_dict[key], set_dict[key]])

            elif settings['set_save_format'] == 'pickle':
                # in this format all data was saved as lists (treat cases where lengths are not equal)
                with open(set_file + '.pickle', 'rb') as fid:
                    set_dict = pickle.load(fid)
                keys = [key for key in set_dict.keys() if '__' not in key]
                for key in keys:
                    if ind_set == 0:
                        data_dict[key] = set_dict[key]
                    else:
                        data_dict[key] += set_dict[key]

            else:
                raise ValueError('invalid set_save_format: ' + str(settings['set_save_format']))

        data_dict['settings'] = settings
        data_dict['field_dict'] = field_dict

        data_dict_file = save_dir + '.pickle'
        with open(compiled_file, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
