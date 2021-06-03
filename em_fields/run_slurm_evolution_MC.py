import os

from slurmpy.slurmpy import Slurm

from em_fields.slurm_functions import get_script_evolution_slave_fenchel2

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel2()

import matplotlib.pyplot as plt

from em_fields.default_settings import define_default_settings, define_default_field

import numpy as np
from scipy.io import savemat
from scipy.stats import maxwell

slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

main_folder = '/home/talm/code/single_particle/slurm_runs/'
main_folder += '/set4/'

plt.close('all')

# define settings
settings = {}

settings = define_default_settings()
field_dict = define_default_field(settings)

# simulation duration
# cyclotron_periods = 1000
cyclotron_periods = int(100 * settings['l'] / settings['v_th'] / field_dict['tau_cyclotron'])
settings['cyclotron_periods'] = cyclotron_periods

save_dir = ''
save_dir += 'tmax_' + str(settings['cyclotron_periods'])
save_dir += '_B0_' + str(field_dict['B0'])
save_dir += '_T_' + str(settings['T_keV'])
save_dir += '_' + str(field_dict['RF_type'])
save_dir += '_ERF_' + str(field_dict['E_RF_kVm'])
save_dir += '_alpha_' + '_'.join([str(alpha_detune) for alpha_detune in field_dict['alpha_detune_list']])
save_dir += '_v2'

print('save_dir: ' + str(save_dir))

settings['save_dir'] = main_folder + '/' + save_dir
os.makedirs(settings['save_dir'], exist_ok=True)
os.chdir(settings['save_dir'])

total_number_of_combinations = 1000

# sampling velocity from Maxwell-Boltzmann
scale = np.sqrt(settings['kB_eV'] * settings['T_eV'] / settings['mi'])
v_abs_samples = maxwell.rvs(size=total_number_of_combinations, scale=scale)

# sampling a random direction
rand_unit_vec = np.random.randn(total_number_of_combinations, 3)
for i in range(total_number_of_combinations):
    rand_unit_vec[i, :] /= np.linalg.norm(rand_unit_vec[i, :])

# total velocity vector
v_0 = rand_unit_vec
for i in range(total_number_of_combinations):
    v_0[i, :] *= v_abs_samples[i]

# create and save the points file to be run later
settings['run_info_file'] = settings['save_dir'] + '/run_info.mat'

mat_dict = {}
mat_dict['v_0'] = v_0
mat_dict['settings'] = settings
mat_dict['field_dict'] = field_dict

savemat(settings['run_info_file'], mat_dict)

# divide the points to a given number of cpus (250 is max in partition core)
num_cpus = 100
num_points_per_cpu = int(np.floor(1.0 * total_number_of_combinations / num_cpus))
num_extra_points = np.mod(total_number_of_combinations, num_cpus)

points_set_list = []
index_first = 0
num_sets = num_cpus if num_points_per_cpu > 0 else num_extra_points
for i in range(num_sets):
    index_last = index_first + num_points_per_cpu
    if i < num_extra_points:
        index_last += 1
    points_set_list += [[k for k in range(index_first, index_last)]]
    index_first = index_last

# run the slave_fenchel scripts on multiple cpus
cnt = 0
for ind_set, points_set in enumerate(points_set_list):
    run_name = 'set_' + str(ind_set) + '_' + save_dir
    print('run_name = ' + run_name)

    settings['ind_set'] = ind_set
    settings['points_set'] = points_set

    command = evolution_slave_fenchel_script \
              + ' --settings "' + str(settings) + '"' \
              + ' --field_dict "' + str(field_dict) + '"'
    s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
    s.run(command)
    cnt += 1
    print('run set # ' + str(cnt) + ' / ' + str(num_sets))
