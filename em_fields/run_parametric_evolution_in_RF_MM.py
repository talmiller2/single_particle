import os
import pickle
import time

import matplotlib.pyplot as plt

from em_fields.default_settings import define_default_settings
from em_fields.em_functions import evolve_particle_in_em_fields, get_thermal_velocity, get_cyclotron_angular_frequency
from em_fields.magnetic_forms import *

start_time = time.time()

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})

plt.close('all')

# save_dir = '../runs/set1/'
save_dir = '../runs/set2/'

os.makedirs(save_dir, exist_ok=True)

# RF_type = 'uniform'
# RF_type_list = ['uniform', 'traveling']
RF_type_list = ['uniform']

# RF definitions
E_RF_kV = 0
# E_RF_kV = 3  # kV/m
E_RF = E_RF_kV * 1e3  # the SI units is V/m

# alpha_detune = 1.1
# alpha_detune = 1.5
# alpha_detune = 2.0
alpha_detune_list = [1.1, 1.5, 2.0]

angle_to_z_axis_list = np.linspace(0, 180, 19)
# angle_to_z_axis_list = np.linspace(0, 180, 5)
# angle_to_z_axis_list = np.linspace(0, 180, 1)

# z0_list = np.linspace(0, 0.5, 6)
# z0_list = np.linspace(0, 0.5, 10)
z0_list = [0.5]

# r0 = 0
# r0 = 0.1
# r0_list = [0, 0.1, 0.2]
r0_list = [0]

num_runs = len(RF_type_list) * len(angle_to_z_axis_list) * len(z0_list) * len(alpha_detune_list) * len(r0_list)
cnt = 0
print('num_runs = ', num_runs)

for RF_type in RF_type_list:
    if RF_type == 'uniform':
        alpha_detune_list_loop = [None]
    else:
        alpha_detune_list_loop = alpha_detune_list

    for alpha_detune in alpha_detune_list_loop:
        for angle_to_z_axis in angle_to_z_axis_list:
            for z0 in z0_list:
                for r0 in r0_list:
                    cnt += 1
                    print('run ' + str(cnt) + '/' + str(num_runs))

                    # run name
                    run_name = 'RF_' + str(RF_type)
                    run_name += '_' + '{:.0f}'.format(E_RF_kV) + '_kV'
                    if RF_type == 'traveling':
                        run_name += '_alpha_detune_' + str(alpha_detune)
                    run_name += '_angle_' + '{:.0f}'.format(angle_to_z_axis)
                    run_name += '_z0_' + '{:.1f}'.format(z0)
                    run_name += '_r0_' + '{:.1f}'.format(r0)
                    print('run_name:', run_name)

                    settings = define_default_settings()
                    m = settings['mi']
                    q = settings['Z_ion'] * settings['e']  # Coulomb
                    # T_eV = 1e3
                    T_eV = 3e3
                    kB_eV = settings['kB_eV']
                    v_th = get_thermal_velocity(T_eV, m, kB_eV)

                    Rm = 2.0
                    # Rm = 3.0
                    # print('Rm = ' + str(Rm))

                    loss_cone_angle = np.arcsin(Rm ** (-0.5)) * 360 / (2 * np.pi)
                    # print('loss_cone_angle = ' + str(loss_cone_angle))

                    c = 3e8
                    # print('v_th / c = ', v_th / c)

                    # B0 = 0 # Tesla
                    # B0 = 0.01  # Tesla
                    B0 = 0.1  # Tesla
                    # B0 = 0.2  # Tesla
                    # B0 = 1.0  # Tesla
                    # B0 = -1.0 # Tesla
                    # B0 = 5.0  # Tesla

                    omega_cyclotron = get_cyclotron_angular_frequency(q, B0, m)
                    tau_cyclotron = 2 * np.pi / omega_cyclotron

                    # print('angle_to_z_axis = ' + str(angle_to_z_axis) + ' degrees')

                    angle_to_z_axis_rad = angle_to_z_axis / 360 * 2 * np.pi
                    v_0 = np.array([0, np.sin(angle_to_z_axis_rad), np.cos(angle_to_z_axis_rad)])
                    # v_0 = np.array([0, 1, 10])
                    # v_0 = np.array([0, 10, 1])
                    # v_0 = np.array([0, 2, 1])
                    # v_0 = np.array([0, 1, 1])
                    # v_0 = np.array([0, 0.5, 1])

                    # normalize the velocity vector
                    v_0 = v_0 / np.linalg.norm(v_0)
                    v_0 *= v_th
                    v_0_norm = np.linalg.norm(v_0)

                    # l = 0.1  # m (interaction length)
                    # l = 0.5  # m (interaction length)
                    # l = 1.0  # m (interaction length)
                    # l = 2.0  # m (interaction length)
                    l = 5.0  # m (interaction length)
                    # l = 10.0  # m (interaction length)
                    # l = 100.0  # m (interaction length)

                    cyclotron_radius = np.linalg.norm(v_0) / omega_cyclotron
                    # print('cyclotron_radius = ' + str(cyclotron_radius) + ' m')
                    # x_0 = np.array([0, 0, 0])
                    # x_0 = cyclotron_radius * np.array([1, 0, 0])
                    # x_0 = cyclotron_radius * np.array([0, 0, 0])
                    # x_0 = np.array([0, 0, 0])
                    # x_0 = np.array([0, 0, 0.5])
                    # x_0 = np.array([0, 0, 0])

                    x_0 = np.array([r0 * l, 0, z0 * l])

                    z_0 = x_0[2]
                    # z_0 = 0

                    v_z = v_0[2]

                    # t_max = l / v_z
                    # t_max = 3 * l / v_z
                    # t_max = 5 * l / v_z
                    # t_max = 10 * l / v_z
                    t_max = 20 * l / v_z
                    # t_max = 30 * l / v_z
                    # t_max = min(t_max, 100 * tau_cyclotron)
                    t_max = np.abs(t_max)

                    # dt = tau_cyclotron / 50 / Rm
                    # dt = tau_cyclotron / 200
                    dt = tau_cyclotron / 20
                    # dt = tau_cyclotron / 10
                    # num_steps = 1000
                    # num_steps = int(t_max / dt)
                    # num_steps = min(num_steps, 10000)
                    # num_steps = min(num_steps, 20000)
                    num_steps = 20000

                    # print('num_steps = ', num_steps)
                    # print('t_max = ', num_steps * dt, 's')

                    if B0 == 0:  # pick a default
                        anticlockwise = 1
                    else:
                        anticlockwise = np.sign(B0)

                    phase_RF = 0
                    # phase_RF = np.pi / 4
                    # phase_RF = np.pi / 2
                    # phase_RF = np.pi
                    # phase_RF = 1.5 * np.pi

                    if RF_type == 'uniform':
                        omega_RF = omega_cyclotron  # resonance
                        # omega = Rm * omega_cyclotron  # resonance

                        k = omega_RF / c

                    elif RF_type == 'traveling':

                        omega_RF = alpha_detune * omega_cyclotron  # resonance

                        # v_RF = alpha_detune / (alpha_detune - 1) * np.abs(v_z)
                        # v_RF = alpha_detune / (alpha_detune - 1) * v_z
                        v_RF = alpha_detune / (alpha_detune - 1) * v_th
                        k = omega_RF / v_RF


                    def E_function(x, t):
                        z = x[2]
                        return E_RF * np.array([anticlockwise * np.sin(k * (z - z_0) - omega_RF * t + phase_RF),
                                                np.cos(k * (z - z_0) - omega_RF * t + phase_RF),
                                                0])


                    def B_function(x, t):
                        use_transverse_fields = True
                        # use_transverse_fields = False

                        # B_mirror = magnetic_field_constant(B0)
                        B_mirror = magnetic_field_logan(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)
                        # B_mirror = magnetic_field_jaeger(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)
                        # B_mirror = magnetic_field_post(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)

                        # B_RF = 1/c * k_hat cross E_RF
                        # https://en.wikipedia.org/wiki/Sinusoidal_plane-wave_solutions_of_the_electromagnetic_wave_equation
                        # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
                        z = x[2]
                        B_RF = E_RF / c * np.array([-np.sign(k) * np.cos(k * (z - z_0) - omega_RF * t + phase_RF),
                                                    np.sign(k) * anticlockwise * np.sin(
                                                        k * (z - z_0) - omega_RF * t + phase_RF),
                                                    0])
                        # B_RF = 0 # test that does not satisfy Maxwell equations.
                        return B_mirror + B_RF


                    hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function,
                                                        num_steps=num_steps, q=q, m=m, return_fields=True)
                    # TODO: save only cyclotron period averages values
                    # TODO: run with E_RF=0 as well for reference.
                    # TODO: document B0, Rm in the name of the dir

                    # add additional info to hist
                    hist['l'] = l
                    hist['T_eV'] = T_eV
                    hist['m'] = m
                    hist['q'] = q
                    hist['v_th'] = v_th
                    hist['B0'] = B0
                    hist['omega_cyclotron'] = omega_cyclotron
                    hist['tau_cyclotron'] = tau_cyclotron
                    hist['cyclotron_radius'] = cyclotron_radius
                    hist['Rm'] = Rm
                    hist['E_RF'] = E_RF
                    hist['phase_RF'] = phase_RF
                    hist['omega_RF'] = omega_RF
                    hist['RF_type'] = RF_type

                    # save run history
                    hist_file = save_dir + '/' + run_name + '.pickle'
                    with open(hist_file, 'wb') as handle:
                        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

end_time = time.time()
run_time = end_time - start_time
print('run_time = ' + str(run_time) + ' sec, ' + str(run_time / 60.0) + ' min.')
