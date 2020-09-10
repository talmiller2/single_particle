import matplotlib.pyplot as plt
import numpy as np

from loop_fields.loop_functions import get_current_loop_magnetic_field_cylindrical
from loop_fields.plot_functions import plot_loop_symbols, plot_magnetic_field_lines

plt.close('all')

# plot loop field test
# dl = 0.05
# z = np.arange(-0.5, 0.5, dl)
# r = np.arange(-2, 2, dl)
dl = 0.1
# dl = 0.02
# dl = 0.01
# r = np.arange(-2, 2, dl)
r = np.arange(-1.1, 1.1, dl)
# z = np.arange(-5, 10, dl)
# z = np.arange(-10, 20, dl)
z = np.arange(-10, 30, dl)

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
# z_MMM_start = z_solenoid_end + solenoid_r / 2
# z_MMM_start = z_solenoid_end + solenoid_r / 4
z_MMM_start = z_solenoid_end
num_main_cell_coils = int(np.round((z_solenoid_end - z1) / (solenoid_r / 2)))
loop_radius_list_main_cell = [1 for i in range(num_main_cell_coils)]
loop_current_list_main_cell = [1 for i in range(num_main_cell_coils)]
z0_list_main_cell = np.ndarray.tolist(np.linspace(z1, z_solenoid_end, num_main_cell_coils))
color_list_main_cell = ['b' for i in range(num_main_cell_coils)]
scale_list_main_cell = [0.5 for i in range(num_main_cell_coils)]

# phase_list = [0, np.pi/4, np.pi/2]
# phase_list = [0, np.pi / 2, np.pi]
phase_list = [0]

for ind_phase, phase in enumerate(phase_list):
    print('ind_phase = ' + str(ind_phase))

    # lambda_MMM = 4.0
    # lambda_MMM = 8.0
    lambda_MMM = 16.0

    # num_MM_coils = 20
    # num_MM_coils = int(np.round((z2 - z_MMM_start) / lambda_MMM) * 10)
    num_MM_coils = max(int(np.round((z2 - z_MMM_start) / lambda_MMM) * 10),
                       int(np.round((z2 - z_MMM_start) / (solenoid_r / 2))))

    loop_radius_list_MM_cell = [1 for i in range(num_MM_coils)]
    color_list_MM_cell = ['r' for i in range(num_MM_coils)]
    # l_cell = np.linspace(0, 2, num_MM_coils)
    z_MMM_array = np.linspace(z_MMM_start, z2, num_MM_coils)
    # loop_current_list_MM_cell = np.ndarray.tolist(l_cell)
    # loop_current_list_MM_cell = np.ndarray.tolist(2 * np.abs(np.sin(l_cell * np.pi - phase)) ** 5)
    # loop_current_list_MM_cell = np.ndarray.tolist(2 * np.abs(np.sin(l_cell * np.pi - phase)) ** 1)
    # loop_current_list_MM_cell = 1 + np.sin(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase)
    # loop_current_list_MM_cell = 1 + 5 + 5 * np.sin(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase)
    loop_current_list_MM_cell = 1 + 2 + 2 * np.sin(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase)
    # loop_current_list_MM_cell = 2 + np.sin(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase)
    # loop_current_list_MM_cell = 2 + np.sin(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase) ** 5
    scale_list_MM_cell = np.ndarray.tolist(0.5 * loop_current_list_MM_cell / np.max(loop_current_list_MM_cell))
    loop_current_list_MM_cell = np.ndarray.tolist(loop_current_list_MM_cell)
    # z0_list_MM_cell = np.ndarray.tolist(np.linspace(z_solenoid_end, z2, num_MM_coils))
    z0_list_MM_cell = np.ndarray.tolist(z_MMM_array)

    # loop_radius_list = loop_radius_list_main_cell
    # loop_current_list = loop_current_list_main_cell
    # z0_list = z0_list_main_cell

    loop_radius_list = loop_radius_list_main_cell + loop_radius_list_MM_cell
    loop_current_list = loop_current_list_main_cell + loop_current_list_MM_cell
    z0_list = z0_list_main_cell + z0_list_MM_cell
    color_list = color_list_main_cell + color_list_MM_cell
    scale_list = scale_list_main_cell + scale_list_MM_cell

    # start_points = None
    n_lines = 20
    start_points = (np.array([z_MMM_start + np.zeros(n_lines), np.linspace(-1, 1, n_lines)]).transpose())

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
        # scale = 0.5 * loop_current
        scale = 0.25 * loop_current
        # scale = loop_current / np.mean(loop_current_list)
        plot_loop_symbols(z0=z0, loop_radius=loop_radius, scale=scale_list[i], color=color_list[i])

    plt.figure(1)
    plt.subplot(len(phase_list), 1, ind_phase + 1)
    plot_magnetic_field_lines(z, r, B_z_tot, B_r_tot, title='B field, MMM phase = ' + str(phase / np.pi) + '$\pi$',
                              start_points=start_points)

    plt.figure(2)
    # plt.plot(z, B_z_tot[int(B_z_tot.shape[0]/2), :], label='MMM phase = ' + str(phase / np.pi) + '$\pi$')
    plt.plot(z, B_z_tot[int(B_z_tot.shape[0] / 2), :] / B_z_tot[int(B_z_tot.shape[0] / 2), int(B_z_tot.shape[1] / 8)],
             label='MMM phase = ' + str(phase / np.pi) + '$\pi$')
    plt.xlabel('z [m]', size=15)
    # plt.ylabel('$B_z$ [T]', size=15)
    plt.ylabel('$B_z/B_{z,0}$', size=15)
    plt.title('Axial magnetic field on axis (normalized)', size=15)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    # calculate current ratio
    I_max = np.max(loop_current_list_MM_cell)
    I_min = np.min(loop_current_list_MM_cell)
    current_ratio = I_max / I_min
    print('current_ratio = ' + str(current_ratio))

    # calculate mirror ratio Rm
    B_z_on_axis = B_z_tot[int(B_z_tot.shape[0] / 2), :]
    # inds_MMM_section = np.where(z > z_MMM_start)[0]
    # inds_MMM_section = np.where(z > 5)[0]
    inds_MMM_section = np.where(np.abs(z - 15) < 5)[0]
    B_max = np.max(B_z_on_axis[inds_MMM_section])
    B_min = np.min(B_z_on_axis[inds_MMM_section])
    Rm = B_max / B_min
    print('Rm = ' + str(Rm))
