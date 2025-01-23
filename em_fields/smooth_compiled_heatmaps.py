import copy
import glob
import os

from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter

smoothing_sigma = 1

save_dir = '/home/talm/code/single_particle/slurm_runs/'
# save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set26_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set28_B0_1T_l_10m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set29_B0_1T_l_3m_Post_Rm_2_first_cell_center_crossing/'
# save_dir += '/set30_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
# save_dir += '/set31_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set32_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set33_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set34_B0_1T_l_3m_Post_Rm_3_intervals/'
# save_dir += '/set35_B0_0.1T_l_1m_Post_Rm_5_intervals/'
# save_dir += '/set36_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set37_B0_1T_l_1m_Post_Rm_3_intervals/'
# save_dir += '/set38_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set39_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set40_B0_1T_l_1m_Logan_Rm_3_intervals_D_T/'
# save_dir += '/set41_B0_1T_l_1m_Post_Rm_3_intervals_D_T_ERF_25/'
# save_dir += '/set45_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set46_B0_2T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set47_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set49_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set54_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
save_dir += '/set55_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

os.chdir(save_dir)

compiled_files = glob.glob('compiled_*.mat')
for compiled_file in compiled_files:
    print('working on', compiled_file)

    # load original
    mat_dict = loadmat(compiled_file)

    # smooth the heatmaps
    mat_dict_smooth = copy.deepcopy(mat_dict)
    for key in mat_dict.keys():
        if '_end' in key:
            # if type(mat_dict[key]) == np.ndarray and len(mat_dict[key].shape) == 2:
            #     print(key, mat_dict[key].shape)
            mat_dict_smooth[key] = gaussian_filter(mat_dict[key], sigma=smoothing_sigma)

    # save
    smooth_compiled_file = 'smooth_' + compiled_file
    savemat(smooth_compiled_file, mat_dict_smooth)
