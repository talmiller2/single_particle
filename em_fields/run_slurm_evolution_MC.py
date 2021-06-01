import os

from slurmpy.slurmpy import Slurm

from em_fields.slurm_functions import get_script_evolution_slave_fenchel2

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel2()

import matplotlib.pyplot as plt

from em_fields.default_settings import define_default_settings
from em_fields.em_functions import get_thermal_velocity, get_cyclotron_angular_frequency

import numpy as np
from scipy.io import savemat
from scipy.stats import maxwell

slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

main_folder = '/home/talm/code/single_particle/slurm_runs/'
main_folder += '/set4/'

plt.close('all')

settings = define_default_settings()
c = settings['c']
m = settings['mi']
q = settings['q']
T_keV = 3.0
T_eV = T_keV * 1e3
kB_eV = settings['kB_eV']
settings['v_th'] = get_thermal_velocity(T_eV, m, kB_eV)

Rm = 3.0
print('Rm = ' + str(Rm))
loss_cone_angle = np.arcsin(Rm ** (-0.5)) * 360 / (2 * np.pi)
print('loss_cone_angle = ' + str(loss_cone_angle))

# define system parameters

l = 10.0  # m (interaction length)
r_0 = 0.0 * l
# r_0 = 0.1 * l
# r_0 = 0.2 * l
# r_0 = 0.3 * l
# z_0 = 0.0 * l
z_0 = 0.5 * l
B0 = 0.1  # Tesla
# B0 = 1.0  # Tesla
omega_cyclotron = get_cyclotron_angular_frequency(q, B0, m)
tau_cyclotron = 2 * np.pi / omega_cyclotron
settings['r_0'] = r_0
settings['z_0'] = z_0
settings['tau_cyclotron'] = tau_cyclotron

# RF definitions
E_RF_kVm = 0  # kV/m
# E_RF_kVm = 1  # kV/m
# E_RF_kVm = 2  # kV/m
# E_RF_kVm = 4  # kV/m
E_RF = E_RF_kVm * 1e3  # the SI units is V/m

if B0 == 0:  # pick a default
    anticlockwise = 1
else:
    anticlockwise = np.sign(B0)

# RF_type = 'uniform'
RF_type = 'traveling'

if RF_type == 'uniform':
    omega = omega_cyclotron  # resonance
    k = omega / c
elif RF_type == 'traveling':
    # alpha_detune_list = [2]
    alpha_detune_list = [2.718]
    # alpha_detune_list = [3.141]
    # alpha_detune_list = [3]
    # alpha_detune_list = [1.5]
    # alpha_detune_list = [2, 4]
    # alpha_detune_list = [2, 3]
    # alpha_detune_list = [2, 2.718]
    # alpha_detune_list = [2.718, 3.141]
    omega = []
    v_RF = []
    k = []
    for alpha_detune in alpha_detune_list:
        omega += [alpha_detune * omega_cyclotron]  # resonance
        v_RF += [alpha_detune / (alpha_detune - 1) * settings['v_th']]
        k += [omega[-1] / v_RF[-1]]

# cyclotron_periods = 1000
cyclotron_periods = int(10 * l / settings['v_th'] / tau_cyclotron)
settings['cyclotron_periods'] = cyclotron_periods

save_dir = ''
# save_dir += 'r0_' + str(r_0 / l)
# save_dir += '_z0_' + str(z_0 / l)
# save_dir += '_tmax_' + str(cyclotron_periods)
save_dir += 'tmax_' + str(cyclotron_periods)
save_dir += '_B0_' + str(B0)
# save_dir += '_Rm_' + str(Rm)
save_dir += '_T_' + str(T_keV)
save_dir += '_' + str(RF_type)
save_dir += '_ERF_' + str(E_RF_kVm)
# save_dir += '_detune_' + str(alpha_detune)
save_dir += '_alpha_' + '_'.join([str(alpha_detune) for alpha_detune in alpha_detune_list])

print('save_dir: ' + str(save_dir))

settings['save_dir'] = main_folder + '/' + save_dir
os.makedirs(settings['save_dir'], exist_ok=True)
os.chdir(settings['save_dir'])

total_number_of_combinations = 1000

# sampling velocity from Maxwell-Boltzmann
m = settings['mi']
T_keV = 3.0
# T_keV = 10.0
T_eV = T_keV * 1e3
kB_eV = settings['kB_eV']
scale = np.sqrt(kB_eV * T_eV / m)
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
mat_dict = {}
mat_dict['v_0'] = v_0

settings['points_file'] = settings['save_dir'] + '/points.mat'
savemat(settings['points_file'], mat_dict)

# divide the points to a given number of cpus (250 is max in partition core)
# num_cpus = 50
num_cpus = 100
# num_cpus = 125
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
    # run_name = 'set_' + str(ind_set)
    run_name = save_dir + '_set_' + str(ind_set)
    print('run_name = ' + run_name)

    settings['points_set'] = points_set

    field_dict = {}
    field_dict['B0'] = B0
    field_dict['Rm'] = Rm
    field_dict['l'] = l
    field_dict['mirror_field_type'] = 'logan'
    # field_dict['mirror_field_type'] = 'const'
    field_dict['E_RF'] = E_RF
    field_dict['anticlockwise'] = anticlockwise
    field_dict['z_0'] = z_0
    field_dict['k'] = k
    field_dict['omega'] = omega
    field_dict['c'] = c

    command = evolution_slave_fenchel_script \
              + ' --settings "' + str(settings) + '"' \
              + ' --field_dict "' + str(field_dict) + '"'
    s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
    s.run(command)
    cnt += 1
    print('set # ' + str(cnt) + ' / ' + str(num_sets))
