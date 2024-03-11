#!/usr/bin/env python3

import argparse
import ast
import pickle

import numpy as np
from scipy.io import savemat, loadmat
from scipy.signal import argrelextrema

from em_fields.RF_field_forms import E_RF_function, B_RF_function
from em_fields.default_settings import define_default_field
from em_fields.em_functions import evolve_particle_in_em_fields
from em_fields.magnetic_forms import get_mirror_magnetic_field

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

# define the file name where the run's data will be saved
if settings['ind_set'] is not None:
    compiled_set_file_name = settings['save_dir'] + '/set_' + str(settings['ind_set'])
else:
    compiled_set_file_name = settings['save_dir'] + '/' + settings['run_name']

sample_keys = ['t', 'z', 'v', 'v_transverse', 'v_axial', 'Bz']
set_data_dict = {}
for key in sample_keys:
    set_data_dict[key] = []

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

    if settings['apply_random_RF_phase'] is True:
        field_dict['phase_RF_addition'] = runs_dict['phase_RF'][0, ind_point]
        field_dict = define_default_field(settings, field_dict=field_dict)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_RF_function, B_RF_function,
                                        q=settings['q'], m=settings['mi'],
                                        field_dict=field_dict, stop_criterion=settings['stop_criterion'],
                                        num_steps=num_steps, t_max=t_max,
                                        number_of_cell_center_crosses=settings['number_of_time_intervals'])

    # save snapshots of key simulation metrics
    if settings['trajectory_save_method'] == 'intervals':
        inds_samples = range(0, len(hist['t']), int(len(hist['t']) / settings['num_snapshots']))

    elif settings['stop_criterion'] == 'first_cell_center_crossing':
        # extract the first and last states of the evolution
        inds_samples = [0, len(hist['t']) - 1]

    elif settings['stop_criterion'] == 'several_cell_center_crossing':
        # extract several cell center crossings
        inds_samples = hist['inds_cell_center_crossing']

    elif settings['trajectory_save_method'] == 'min_Bz':
        Bz = hist['B'][:, 2]
        inds_samples = argrelextrema(abs(Bz - field_dict['B0']), np.less)[0]

    elif settings['trajectory_save_method'] in ['min_Bz_mirror', 'min_Bz_mirror_const_vz_sign']:

        # pick the indices where the magnetic field crosses the minimum
        B_mirror = []
        for x_curr, t_curr in zip(hist['x'], hist['t']):
            B_mirror += [get_mirror_magnetic_field(x_curr, field_dict['B0'], field_dict['Rm'], field_dict['l'],
                                                   mirror_field_type=field_dict['mirror_field_type'])]
        Bz_mirror = np.array(B_mirror)[:, 2]
        inds_Bz_mirror_extrema = list(argrelextrema(abs(Bz_mirror - field_dict['B0']), np.less)[0])
        inds_Bz_mirror_extrema.insert(0, 0)  # add the initial point

        # filter to only look for the extrema near the points where the magnetic field is close to minimum anyway,
        # to cancel possible glitches in the argrelextrema function
        inds_B_close_to_min = np.where(abs((Bz_mirror - field_dict['B0']) / field_dict['B0']) < 0.05)[0]

        if settings['trajectory_save_method'] == 'min_B_mirror':
            # combine the conditions
            inds_samples = list(set(inds_Bz_mirror_extrema) & set(inds_B_close_to_min))  # combine all conditions
            inds_samples.sort()  # to make the lists monotonic with evolution times

        elif settings['trajectory_save_method'] == 'min_B_mirror_const_vz_sign':
            # pick the indices
            vz = hist['v'][:, 2]
            vz_0 = v_0[2]
            inds_const_vz_sign = np.where(np.sign(vz) == np.sign(vz_0))[0]

            # combine the conditions
            inds_samples = list(
                set(inds_Bz_mirror_extrema) & set(inds_B_close_to_min) & set(
                    inds_const_vz_sign))  # combine all conditions
            inds_samples.sort()  # to make the lists monotonic with evolution times

    else:
        raise ValueError('invalid option for trajectory_save_method: ' + str(settings['trajectory_save_method']))

    # sample the trajectory
    curr_data_dict = {}
    for key in sample_keys:
        curr_data_dict[key] = []

    for i in inds_samples:
        # time
        curr_data_dict['t'] += [hist['t'][i]]

        # axial position
        z = hist['x'][i, 2]
        curr_data_dict['z'] += [z]

        # velocity (total)
        v_abs = np.linalg.norm(hist['v'][i])
        curr_data_dict['v'] += [v_abs]

        # velocity (transverse)
        v_transverse_abs = np.linalg.norm(hist['v'][i, 0:2])
        curr_data_dict['v_transverse'] += [v_transverse_abs]

        # velocity (axial)
        v_axial = hist['v'][i, 2]
        curr_data_dict['v_axial'] += [v_axial]

        # magnetic field (axial)
        Bz = hist['B'][i, 2]
        curr_data_dict['Bz'] += [Bz]

    # combine this point run to larger data dict
    for key in sample_keys:
        set_data_dict[key] += [curr_data_dict[key]]

if settings['set_save_format'] == 'mat':
    savemat(compiled_set_file_name + '.mat', set_data_dict)
elif settings['set_save_format'] == 'pickle':
    with open(compiled_set_file_name + '.pickle', 'wb') as handle:
        pickle.dump(set_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    raise ValueError('invalid set_save_format: ' + str(settings['set_save_format']))
