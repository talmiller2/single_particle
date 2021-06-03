#!/usr/bin/env python3

import argparse
import ast

import numpy as np
from scipy.io import savemat, loadmat

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
mat_dict = loadmat(settings['run_info_file'])

# define the mat_dict where all data will be compiled
compiled_set_file = settings['save_dir'] + '/set_' + str(settings['ind_set']) + '.mat'
set_mat_dict = {}
set_mat_dict['z'] = []
set_mat_dict['E'] = []
set_mat_dict['E_transverse'] = []

# loop over points for current process
for ind_point in settings['points_set']:
    print('ind_point: ' + str(ind_point))

    run_name = 'ind_' + str(ind_point)
    print('run_name = ' + run_name)

    # initial location and velocity of particle
    x_0 = np.array([settings['r_0'], 0, settings['z_0']])
    v_0 = mat_dict['v_0'][ind_point]

    t_max = settings['sim_cyclotron_periods'] * field_dict['tau_cyclotron']
    dt = field_dict['tau_cyclotron'] / 20
    num_steps = int(t_max / dt)

    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_RF_function, B_RF_function,
                                        num_steps=num_steps, q=settings['q'], m=settings['mi'], field_dict=field_dict)

    # save snapshots of key simulation metrics
    num_snapshots = 30
    t_array = []
    z_array = []
    E_array = []
    # E_transverse_array = []
    v_array = []
    v_transverse_array = []
    v_axial_array = []
    for i in range(0, num_steps, int(num_steps / num_snapshots)):
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

        # # energy (total)
        # v_abs = np.linalg.norm(hist['v'][i])
        # E = 0.5 * settings['mi'] * v_abs ** 2.0
        # E_array += [E]
        #
        # # energy (transverse)
        # v_transverse_abs = np.linalg.norm(hist['v'][i, 0:2])
        # E_transverse = 0.5 * settings['mi'] * v_transverse_abs ** 2.0
        # E_transverse_array += [E_transverse]

    # save results of point to file
    # save_array = np.array([z_array, E_array, E_transverse_array])
    save_array = np.array([z_array, v_array, v_transverse_array, v_axial_array])
    save_file_path = settings['save_dir'] + '/' + run_name + '.txt'
    np.savetxt(save_file_path, save_array)

    # perform compilation of results at the process level as well to make it faster after
    set_mat_dict['z'] += [z_array]
    # set_mat_dict['E'] += [E_array]
    # set_mat_dict['E_transverse'] += [E_transverse_array]
    set_mat_dict['v'] += [v_array]
    set_mat_dict['v_transverse'] += [v_transverse_array]
    set_mat_dict['v_axial'] += [v_axial_array]

set_mat_dict['t'] = t_array
savemat(compiled_set_file, set_mat_dict)
