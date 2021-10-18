#!/usr/bin/env python3

import argparse
import ast
import pickle

import numpy as np
from scipy.io import savemat, loadmat
# from scipy.signal import find_peaks # only supported above scipy 1.1
from scipy.signal import argrelextrema

from em_fields.RF_field_forms import E_RF_function, B_RF_function
from em_fields.em_functions import evolve_particle_in_em_fields

parser = argparse.ArgumentParser()
parser.add_argument('--settings', help='settings (dict) for the maxwell simulation',
                    type=str, required=True)
parser.add_argument('--field_dict', help='field_dict do define the field the particle will experience',
                    type=str, required=True)

args = parser.parse_args()
print('args.settings = ' + str(args.settings))
settings = ast.literal_eval(args.settings)
print('args.field_dict = ' + str(args.field_dict))
field_dict = ast.literal_eval(args.field_dict)

# load data for runs
runs_dict_file = settings['save_dir'] + '/points_dict.mat'
runs_dict = loadmat(runs_dict_file)

# define the construct where this set's data will be saved
compiled_set_file_without_suffix = settings['save_dir'] + '/set_' + str(settings['ind_set'])

set_data_dict = {}
set_data_dict['t'] = []
set_data_dict['z'] = []
set_data_dict['v'] = []
set_data_dict['v_transverse'] = []
set_data_dict['v_axial'] = []
set_data_dict['Bz'] = []

# loop over points for current process
for ind_point in settings['points_set']:
    print('ind_point: ' + str(ind_point))

    run_name = 'ind_' + str(ind_point)
    print('run_name = ' + run_name)

    # initial location and velocity of particle
    x_0 = np.array([settings['r_0'], 0, settings['z_0']])
    v_0 = runs_dict['v_0'][ind_point]

    t_max = settings['sim_cyclotron_periods'] * field_dict['tau_cyclotron']
    dt = field_dict['tau_cyclotron'] / settings['time_step_tau_cyclotron_divisions']
    num_steps = int(t_max / dt)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_RF_function, B_RF_function,
                                        num_steps=num_steps, q=settings['q'], m=settings['mi'], field_dict=field_dict)

    # save snapshots of key simulation metrics
    if settings['trajectory_save_method'] == 'intervals':
        inds_samples = range(0, num_steps, int(num_steps / settings['num_snapshots']))
    elif settings['trajectory_save_method'] == 'min_B':
        Bz = hist['B'][:, 2]
        # inds_samples = find_peaks(-abs(Bz - field_dict['B0']))[0] # only supported above scipy 1.1
        inds_samples = argrelextrema(abs(Bz - field_dict['B0']), np.less)[0]
    else:
        raise ValueError('invalid option for trajectory_save_method: ' + str(settings['trajectory_save_method']))

    # sample the trajectory
    t_array = []
    z_array = []
    v_array = []
    v_transverse_array = []
    v_axial_array = []
    Bz_array = []

    for i in inds_samples:
        # time
        t_array += [hist['t'][i]]

        # axial position
        z = hist['x'][i, 2]
        z_array += [z]

        # velocity (total)
        v_abs = np.linalg.norm(hist['v'][i])
        v_array += [v_abs]

        # velocity (transverse)
        v_transverse_abs = np.linalg.norm(hist['v'][i, 0:2])
        v_transverse_array += [v_transverse_abs]

        # velocity (axial)
        v_axial = hist['v'][i, 2]
        v_axial_array += [v_axial]

        # magnetic field (axial)
        Bz = hist['B'][i, 2]
        Bz_array += [Bz]

    # perform compilation of results at the process level as well to make it faster after
    set_data_dict['t'] += [t_array]
    set_data_dict['z'] += [z_array]
    set_data_dict['v'] += [v_array]
    set_data_dict['v_transverse'] += [v_transverse_array]
    set_data_dict['v_axial'] += [v_axial_array]
    set_data_dict['Bz'] += [Bz_array]

if settings['set_save_format'] == 'mat':
    savemat(compiled_set_file_without_suffix + '.mat', set_data_dict)
elif settings['set_save_format'] == 'pickle':
    with open(compiled_set_file_without_suffix + '.pickle', 'wb') as handle:
        pickle.dump(set_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    raise ValueError('invalid set_save_format: ' + str(settings['set_save_format']))
