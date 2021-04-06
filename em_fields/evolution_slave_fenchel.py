#!/usr/bin/env python3

import argparse
import ast

import numpy as np
from scipy.io import loadmat

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

# load the points to run
mat_dict = loadmat(settings['points_file'])

# loop over points for current process
for ind_point in settings['points_set']:
    print('ind_point: ' + str(ind_point))

    # extract the parameters that vary
    settings['v_abs'] = settings['v_th'] * mat_dict['v_abs'][0][ind_point]
    settings['angle_to_z_axis'] = mat_dict['angle_to_z_axis'][0][ind_point]
    field_dict['phase_RF'] = mat_dict['phase_RF'][0][ind_point]
    print('v_abs: ' + str(settings['v_abs']))
    print('angle_to_z_axis: ' + str(settings['angle_to_z_axis']))
    print('phase_RF: ' + str(field_dict['phase_RF']))

    run_name = ''
    run_name += 'v_' + '{:.2f}'.format(settings['v_abs'] / settings['v_th'])
    run_name += '_angle_' + str(int(settings['angle_to_z_axis']))
    run_name += '_phaseRF_' + '{:.2f}'.format(field_dict['phase_RF'] / np.pi)
    print('run_name = ' + run_name)

    # initial location and velocity of particle
    x_0 = np.array([settings['r_0'], 0, settings['z_0']])
    angle_to_z_axis_rad = settings['angle_to_z_axis'] / 360 * 2 * np.pi
    v_0 = settings['v_abs'] * np.array([0, np.sin(angle_to_z_axis_rad), np.cos(angle_to_z_axis_rad)])

    t_max = settings['cyclotron_periods'] * settings['tau_cyclotron']
    dt = settings['tau_cyclotron'] / 100
    num_steps = int(t_max / dt)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_RF_function, B_RF_function,
                                        num_steps=num_steps, q=settings['q'], m=settings['mi'], field_dict=field_dict)
    z_end = hist['x'][:, 2][-1]
    print('z_end = ' + str(z_end))

    v_abs_fin = np.linalg.norm(hist['v'][-1])
    energy_end = (v_abs_fin / settings['v_abs']) ** 2.0
    print('energy_end = ' + str(energy_end))

    # save results to file
    save_array = np.array([z_end, energy_end])
    save_file_path = settings['save_dir'] + '/' + run_name + '.txt'
    np.savetxt(save_file_path, save_array)