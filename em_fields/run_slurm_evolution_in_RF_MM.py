import os

from slurmpy.slurmpy import Slurm

from em_fields.slurm_functions import get_script_evolution_slave

pwd = os.getcwd()
evolution_slace_script = get_script_evolution_slave()

import matplotlib.pyplot as plt

from em_fields.default_settings import define_default_settings
from em_fields.em_functions import get_thermal_velocity, get_cyclotron_angular_frequency

import numpy as np

slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

main_folder = '/home/talm/code/single_particle/slurm_runs/set1/'

plt.close('all')

settings = define_default_settings()
c = settings['c']
m = settings['mi']
q = settings['Z_ion'] * settings['e']  # Coulomb
T_keV = 3.0
T_eV = T_keV * 1e3
kB_eV = settings['kB_eV']
v_th = get_thermal_velocity(T_eV, m, kB_eV)

Rm = 3.0
print('Rm = ' + str(Rm))
loss_cone_angle = np.arcsin(Rm ** (-0.5)) * 360 / (2 * np.pi)
print('loss_cone_angle = ' + str(loss_cone_angle))

# define system parameters

l = 10.0  # m (interaction length)
r_0 = 0
z_0 = 0.5 * l
B0 = 0.1  # Tesla
omega_cyclotron = get_cyclotron_angular_frequency(q, B0, m)
tau_cyclotron = 2 * np.pi / omega_cyclotron
settings['r_0'] = r_0
settings['z_0'] = z_0

# RF definitions
E_RF_kVm = 2  # kV/m
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
    alpha_detune = 2
    omega = alpha_detune * omega_cyclotron  # resonance
    v_RF = alpha_detune / (alpha_detune - 1) * v_th
    k = omega / v_RF

cyclotron_periods = 1000

save_dir = ''
save_dir += 'tmax_' + str(cyclotron_periods)
save_dir += 'B0_' + str(B0)
save_dir += '_Rm_' + str(Rm)
save_dir += '_T_' + str(T_keV)
save_dir += '_' + str(RF_type)
save_dir += '_ERF_' + str(E_RF_kVm)
save_dir += '_detune_' + str(alpha_detune)

# v_abs_list = v_th * np.linspace(0.5, 1.5, 11)
# angle_to_z_axis_list = [i for i in range(0, 181, 5)]
# phase_RF_list = np.array([0, 0.25, 0.5]) * np.pi

v_abs_list = v_th * np.linspace(0.5, 1.5, 3)
angle_to_z_axis_list = [i for i in range(0, 181, 30)]
phase_RF_list = np.array([0]) * np.pi

total_number_of_combinations = 1
total_number_of_combinations *= len(v_abs_list)
total_number_of_combinations *= len(angle_to_z_axis_list)
total_number_of_combinations *= len(phase_RF_list)
cnt = 1

settings['save_dir'] = main_folder
print('save_dir: ' + save_dir)

for v_abs in v_abs_list:
    for angle_to_z_axis in angle_to_z_axis_list:
        for phase_RF in phase_RF_list:
            run_name = ''
            run_name += 'v_' + '{:.1f}'.format(v_abs / v_th)
            run_name += '_angle_' + str(angle_to_z_axis)
            run_name += '_phaseRF_' + '{:.2f}'.format(phase_RF / np.pi)
            print('run_name = ' + run_name)

            settings['v_abs'] = v_abs
            settings['angle_to_z_axis'] = angle_to_z_axis

            field_dict = {}
            field_dict['B0'] = B0
            field_dict['Rm'] = Rm
            field_dict['l'] = l
            field_dict['mirror_field_type'] = 'logan'
            field_dict['E_RF'] = E_RF
            field_dict['anticlockwise'] = anticlockwise
            field_dict['z_0'] = z_0
            field_dict['k'] = k
            field_dict['omega'] = omega
            field_dict['phase_RF'] = phase_RF
            field_dict['c'] = c

            # settings['save_dir'] = main_folder + '/' + run_name
            # print('save dir: ' + str(settings['save_dir']))
            settings['save_dir'] = main_folder
            os.makedirs(settings['save_dir'], exist_ok=True)
            os.chdir(settings['save_dir'])

            command = evolution_slace_script \
                      + ' --settings "' + str(settings) + '"' \
                      + ' --field_dict "' + str(field_dict) + '"'
            s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
            s.run(command)
            cnt += 1
            print('run # ' + str(cnt) + ' / ' + str(total_number_of_combinations))

            os.chdir(pwd)