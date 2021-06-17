import os
import pickle

from scipy.stats import maxwell
from slurmpy.slurmpy import Slurm

from em_fields.slurm_functions import get_script_evolution_slave_fenchel2

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel2()

import matplotlib.pyplot as plt

from em_fields.default_settings import define_default_settings, define_default_field

import numpy as np
from scipy.io import savemat

slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

main_folder = '/home/talm/code/single_particle/slurm_runs/'
main_folder += '/set5/'
# main_folder += '/set6/'

plt.close('all')

# define settings
settings = {}

settings = define_default_settings()

field_dict = {}

# field_dict['E_RF_kVm'] = 0  # kV/m
field_dict['E_RF_kVm'] = 1  # kV/m
# field_dict['E_RF_kVm'] = 3  # kV/m
# field_dict['E_RF_kVm'] = 5  # kV/m
# field_dict['E_RF_kVm'] = 10  # kV/m

# field_dict['v_z_factor_list'] = [1]
field_dict['v_z_factor_list'] = [2]
# field_dict['v_z_factor_list'] = [1, 2]
# field_dict['v_z_factor_list'] = [1, 1.5, 2]

# field_dict['alpha_detune_list'] = [1 for i in range(len(field_dict['v_z_factor_list']))]
field_dict['alpha_detune_list'] = [1.1 for i in range(len(field_dict['v_z_factor_list']))]
# field_dict['alpha_detune_list'] = [1.3 for i in range(len(field_dict['v_z_factor_list']))]
# field_dict['alpha_detune_list'] = [1.7 for i in range(len(field_dict['v_z_factor_list']))]
# field_dict['alpha_detune_list'] = [2 for i in range(len(field_dict['v_z_factor_list']))]

# field_dict['nullify_RF_magnetic_field'] = True

field_dict = define_default_field(settings, field_dict=field_dict)

# simulation duration
sim_cyclotron_periods = int(20 * settings['l'] / settings['v_th'] / field_dict['tau_cyclotron'])
settings['sim_cyclotron_periods'] = sim_cyclotron_periods

save_dir = ''
save_dir += 'tmax_' + str(settings['sim_cyclotron_periods'])
save_dir += '_B0_' + str(field_dict['B0'])
save_dir += '_T_' + str(settings['T_keV'])
# save_dir += '_nonMB'
# save_dir += '_' + str(field_dict['RF_type'])
if field_dict['E_RF_kVm'] > 0:
    save_dir += '_ERF_' + str(field_dict['E_RF_kVm'])
    save_dir += '_alpha_' + '_'.join([str(alpha_detune) for alpha_detune in field_dict['alpha_detune_list']])
    save_dir += '_vz_' + '_'.join([str(v_z_factor) for v_z_factor in field_dict['v_z_factor_list']])
if field_dict['nullify_RF_magnetic_field']:
    save_dir += '_zeroBRF'

print('save_dir: ' + str(save_dir))

settings['save_dir'] = main_folder + '/' + save_dir
os.makedirs(settings['save_dir'], exist_ok=True)
os.chdir(settings['save_dir'])

settings_file = settings['save_dir'] + '/settings.pickle'
with open(settings_file, 'wb') as handle:
    pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
field_dict_file = settings['save_dir'] + '/field_dict.pickle'
with open(field_dict_file, 'wb') as handle:
    pickle.dump(field_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# total_number_of_combinations = 20000
total_number_of_combinations = 1000

# sampling velocity from Maxwell-Boltzmann
scale = np.sqrt(settings['kB_eV'] * settings['T_eV'] / settings['mi'])
v_abs_samples = maxwell.rvs(size=total_number_of_combinations, scale=scale)
# v_abs_samples = settings['v_th'] * np.ones(total_number_of_combinations)  # testing constant velocity

# sampling a random direction
rand_unit_vec = np.random.randn(total_number_of_combinations, 3)
for i in range(total_number_of_combinations):
    rand_unit_vec[i, :] /= np.linalg.norm(rand_unit_vec[i, :])

# sampling a random direction but only within the right-LC
u = np.random.rand(total_number_of_combinations)
v = np.random.rand(total_number_of_combinations)
theta_max = settings['loss_cone_angle'] / 360 * 2 * np.pi
v_min = (np.cos(theta_max) + 1) / 2
v *= (1 - v_min)
v += v_min
phi = 2 * np.pi * u  # longitude
theta = np.arccos(2 * v - 1)  # latitude
x = np.cos(phi) * np.sin(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(theta)
rand_unit_vec = np.array([x, y, z]).T

# total velocity vector
v_0 = rand_unit_vec
for i in range(total_number_of_combinations):
    v_0[i, :] *= v_abs_samples[i]

# create and save the points file to be run later
runs_dict = {'v_0': v_0}
runs_dict_file = settings['save_dir'] + '/runs_dict.mat'
savemat(runs_dict_file, runs_dict)

# divide the points to a given number of cpus (250 is max in partition core)
num_cpus = 50
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
    print('run set # ' + str(cnt) + ' / ' + str(num_sets))
    cnt += 1
