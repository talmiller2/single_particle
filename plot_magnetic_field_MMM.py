import matplotlib.pyplot as plt
import numpy as np

from magnetic_field_functions import get_current_loop_magnetic_field_cylindrical
from plot_functions import plot_loop_symbols, plot_magnetic_field_lines

plt.close('all')

# plot loop field test
# dl = 0.05
# z = np.arange(-0.5, 0.5, dl)
# r = np.arange(-2, 2, dl)
# dl = 0.1
dl = 0.02
# dl = 0.01
# r = np.arange(-2, 2, dl)
r = np.arange(-1.1, 1.1, dl)
z = np.arange(-5, 10, dl)

rr, zz = np.meshgrid(r, z, indexing='ij')
# print('rr shape = ' + str(rr.shape))
# print('zz shape = ' + str(zz.shape))

n_r = len(r)
n_z = len(z)

### Main section + MM

z1 = min(z)
z2 = max(z)
z_solenoid_end = z1 + 0.4 * (z2 - z1)
# num_main_cell_coils = 10
solenoid_r = 1.0
z_MMM_start = z_solenoid_end + solenoid_r / 2
num_main_cell_coils = int(np.round((z_solenoid_end - z1) / (solenoid_r / 2)))
loop_radius_list_main_cell = [1 for i in range(num_main_cell_coils)]
loop_current_list_main_cell = [1 for i in range(num_main_cell_coils)]
z0_list_main_cell = np.ndarray.tolist(np.linspace(z1, z_solenoid_end, num_main_cell_coils))

# phase_list = [0, np.pi/4, np.pi/2]
phase_list = [0, np.pi / 2, np.pi]

for ind_phase, phase in enumerate(phase_list):

    lambda_MMM = 4.0
    # phase = 0
    # phase = np.pi / 2
    num_MM_coils = 20
    loop_radius_list_MM_cell = [1 for i in range(num_MM_coils)]
    # l_cell = np.linspace(0, 2, num_MM_coils)
    z_MMM_array = np.linspace(z_MMM_start, z2, num_MM_coils)
    # loop_current_list_MM_cell = np.ndarray.tolist(l_cell)
    # loop_current_list_MM_cell = np.ndarray.tolist(2 * np.abs(np.sin(l_cell * np.pi - phase)) ** 5)
    # loop_current_list_MM_cell = np.ndarray.tolist(2 * np.abs(np.sin(l_cell * np.pi - phase)) ** 1)
    loop_current_list_MM_cell = np.ndarray.tolist(
        1 + np.sin(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase))
    # z0_list_MM_cell = np.ndarray.tolist(np.linspace(z_solenoid_end, z2, num_MM_coils))
    z0_list_MM_cell = np.ndarray.tolist(z_MMM_array)

    # loop_radius_list = loop_radius_list_main_cell
    # loop_current_list = loop_current_list_main_cell
    # z0_list = z0_list_main_cell

    loop_radius_list = loop_radius_list_main_cell + loop_radius_list_MM_cell
    loop_current_list = loop_current_list_main_cell + loop_current_list_MM_cell
    z0_list = z0_list_main_cell + z0_list_MM_cell

    ######################
    B_r_tot, B_z_tot = 0, 0

    for i in range(len(loop_radius_list)):
        loop_radius = loop_radius_list[i]
        loop_current = loop_current_list[i]
        z0 = z0_list[i]

        B_r, B_z = get_current_loop_magnetic_field_cylindrical(rr, zz, z0=z0, loop_radius=loop_radius,
                                                               loop_current=loop_current)

        B_r_tot += B_r
        B_z_tot += B_z

        plt.figure(1)
        plt.subplot(len(phase_list), 1, ind_phase + 1)

        # scale = 1.0
        scale = 0.5 * loop_current
        # scale = loop_current / np.mean(loop_current_list)
        plot_loop_symbols(z0=z0, loop_radius=loop_radius, scale=scale)

    plt.figure(1)
    plt.subplot(len(phase_list), 1, ind_phase + 1)
    plot_magnetic_field_lines(z, r, B_z_tot, B_r_tot, title='B field, MMM phase = ' + str(phase / np.pi) + '$\pi$')
