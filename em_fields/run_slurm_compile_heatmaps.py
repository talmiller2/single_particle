import os

import numpy as np
from slurmpy.slurmpy import Slurm

from em_fields.slurm_functions import get_compile_heatmap_slave

compile_heatmap_slave_script = get_compile_heatmap_slave()

slurm_kwargs = {}
slurm_kwargs['partition'] = 'core'
# slurm_kwargs['partition'] = 'testCore'
# slurm_kwargs['partition'] = 'socket'
# slurm_kwargs['partition'] = 'testSocket'
slurm_kwargs['ntasks'] = 1
# slurm_kwargs['cpus-per-task'] = 1
# slurm_kwargs['cores-per-socket'] = 1

local = False
# local = True

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
# save_dir += '/set51_B0_1T_l_1m_Post_Rm_6_intervals_D_T/'
# save_dir += '/set52_B0_1T_l_2m_Post_Rm_3_intervals_D_T/'
# save_dir += '/set54_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
save_dir += '/set55_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

use_RF = True
# use_RF = False

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'

radial_distribution = 'uniform'

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
# beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.6, 1.0, 21), 2)  # set28
# alpha_loop_list = alpha_loop_list[10::]
# beta_loop_list = np.round(np.linspace(-5, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 21), 2)  # set29, set30
# beta_loop_list = np.round(np.linspace(-10, 0, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 11), 2)  # set31, 32, 33
# beta_loop_list = np.round(np.linspace(-10, 0, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set34
# beta_loop_list = np.round(np.linspace(-5, 5, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.0, 5), 2)  # set35
# beta_loop_list = np.round(np.linspace(-10, 0, 5), 2)

# alpha_loop_list = np.round(np.linspace(0.8, 1.2, 21), 2)  # set36
# beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.5, 1.5, 21), 2)  # set37, 39, 40
# beta_loop_list = np.round(np.linspace(-10, 10, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 21), 2)  # set38
# beta_loop_list = np.round(np.linspace(-5, 5, 21), 2)

# alpha_loop_list = np.round(np.linspace(0.7, 1.3, 11), 2)  # set43
# beta_loop_list = np.round(np.linspace(-2, 2, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.4, 1.6, 21), 2)  # set47, 49, 50
# beta_loop_list = np.round(np.linspace(-2, 2, 21), 2)

alpha_loop_list = np.round(np.linspace(0.4, 1.6, 11), 2)  # set51, 52, 55
beta_loop_list = np.round(np.linspace(-2, 2, 11), 2)

# alpha_loop_list = np.round(np.linspace(0.4, 1.6, 5), 2)  # set54
# beta_loop_list = np.round(np.linspace(-2, 2, 5), 2)


gas_name_list = ['deuterium', 'tritium']

RF_type_list = []
RF_amplitude_list = []
RF_type_list += ['magnetic_transverse', 'magnetic_transverse']
RF_amplitude_list += [0.02, 0.04]  # [T]
RF_type_list += ['electric_transverse', 'electric_transverse']
RF_amplitude_list += [25, 50]  # kV/m

# sigma_r0_list = [0.1]
sigma_r0_list = [0.05]
induced_fields_factor_list = [1, 0]
# with_kr_correction_list = [False, True]
with_kr_correction_list = [True]
time_step_tau_cyclotron_divisions = 50

# theta_type_list = ['sign_vz0', 'sign_vz']
theta_type_list = ['sign_vz']

os.chdir(save_dir)

for theta_type in theta_type_list:
    for RF_type, RF_amplitude in zip(RF_type_list, RF_amplitude_list):
        for gas_name in gas_name_list:
            for sigma_r0 in sigma_r0_list:
                for induced_fields_factor in induced_fields_factor_list:
                    for with_kr_correction in with_kr_correction_list:

                        ## save compiled data to file
                        set_name = 'compiled_'
                        set_name += theta_type + '_'
                        if RF_type == 'electric_transverse':
                            set_name += 'ERF_' + str(RF_amplitude)
                        elif RF_type == 'magnetic_transverse':
                            set_name += 'BRF_' + str(RF_amplitude)
                        if induced_fields_factor < 1.0:
                            set_name += '_iff' + str(induced_fields_factor)
                        if with_kr_correction == True:
                            set_name += '_withkrcor'
                        set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
                        if absolute_velocity_sampling_type == 'const_vth':
                            set_name += '_const_vth'
                        if sigma_r0 > 0:
                            set_name += '_sigmar' + str(sigma_r0)
                            if radial_distribution == 'normal':
                                set_name += 'norm'
                            elif radial_distribution == 'uniform':
                                set_name += 'unif'
                        set_name += '_' + gas_name
                        compiled_save_file = save_dir + '/' + set_name + '.mat'
                        print('****** compiled_save_file', compiled_save_file)

                        # collect all necessary info in a dict to be passed on
                        settings = {}
                        settings['use_RF'] = use_RF
                        settings['alpha_loop_list'] = list(alpha_loop_list)
                        settings['beta_loop_list'] = list(beta_loop_list)
                        settings['save_dir'] = save_dir
                        settings['set_name'] = set_name
                        settings['RF_type'] = RF_type
                        settings['RF_amplitude'] = RF_amplitude
                        settings['induced_fields_factor'] = induced_fields_factor
                        settings['with_kr_correction'] = with_kr_correction
                        settings['time_step_tau_cyclotron_divisions'] = time_step_tau_cyclotron_divisions
                        settings['absolute_velocity_sampling_type'] = absolute_velocity_sampling_type
                        settings['sigma_r0'] = sigma_r0
                        settings['radial_distribution'] = radial_distribution
                        settings['gas_name'] = gas_name
                        settings['compiled_save_file'] = compiled_save_file

                        #  checking if the save file already exists
                        if os.path.exists(compiled_save_file):
                            print('already exists, not running.')
                        else:
                            command = compile_heatmap_slave_script + ' --settings "' + str(settings) + '"'
                            slurm_run_name = 'compile_' + set_name
                            print('running', slurm_run_name)
                            s = Slurm(slurm_run_name, slurm_kwargs=slurm_kwargs)
                            s.run(command, local=local)
