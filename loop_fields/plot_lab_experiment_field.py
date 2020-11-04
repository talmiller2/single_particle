import matplotlib.pyplot as plt
import numpy as np

from loop_fields.loop_functions import get_current_loop_magnetic_field_cylindrical
from loop_fields.plot_functions import plot_loop_symbols, plot_magnetic_field_lines

plt.rcParams.update({'font.size': 15})
plt.close('all')

plot_B_map = True
# plot_B_map = False

I_cube_DC = 40e3  # A

# I_tube_AC = 0  # A
# I_tube_AC = 40e3  # A

I_tube_DC = 50e3  # A
I_tube_AC = 50e3  # A

# I_tube_DC = 40e3  # A
# I_tube_AC = 40e3  # A

# I_tube_DC = 70e3  # A
# I_tube_AC = 100e3  # A

phase = 0
# phase = 0.01 * np.pi
# phase = 0.1 * np.pi
# phase = 0.5 * np.pi
# phase = np.pi
# phase = 1.5 * np.pi
# phase = 2.0 * np.pi

r_cube_DC = 9.1e-2  # m
r_tube_AC = 3.8e-2  # m
r_tube_DC = 4.8e-2  # m

z1_cube = -3 * r_cube_DC
z2_cube = 0

wavelength = 15e-2  # m
# wavelength = 30e-2  # m
# wavelength = 10e-2  # m
num_cells = 5
z1_tube = r_tube_AC
z2_tube = z1_tube + num_cells * wavelength

# define a grid
dr = 0.1e-2
r = np.arange(-1.5 * r_tube_DC, 1.5 * r_tube_DC, dr)
# r = np.array([0])
r_cm = r * 1e2

dl = wavelength / 50
z_calc_B_1 = -2 * r_cube_DC
z_calc_B_2 = (num_cells - 1) * wavelength
# z = np.arange(z1_cube, z2_tube, dl)
z = np.arange(z_calc_B_1, z_calc_B_2, dl)
z_cm = z * 1e2
rr, zz = np.meshgrid(r, z, indexing='ij')

n_r = len(r)
n_z = len(z)

# define all coils
loop_radius_list = []
loop_current_list = []
z0_list = []
color_list = []
scale_list = []

## cube section ##
num_cube_coils = int(np.round((z2_cube - z1_cube) / r_cube_DC)) + 1
loop_radius_list += [r_cube_DC for i in range(num_cube_coils)]
loop_current_list += [I_cube_DC for i in range(num_cube_coils)]
z0_list += np.ndarray.tolist(np.linspace(z1_cube, z2_cube, num_cube_coils))
color_list += ['g' for i in range(num_cube_coils)]
scale_list += [0.5 for i in range(num_cube_coils)]

### MMM section DC coils ###
num_tube_DC_coils = int(np.round((z2_tube - z1_tube) / r_tube_DC)) + 1
loop_radius_list += [r_tube_DC for i in range(num_tube_DC_coils)]
loop_current_list += [I_tube_DC for i in range(num_tube_DC_coils)]
z0_list += np.ndarray.tolist(np.linspace(z1_tube, z2_tube, num_tube_DC_coils))
color_list += ['g' for i in range(num_tube_DC_coils)]
scale_list += [0.5 for i in range(num_tube_DC_coils)]

### MMM section AC coils ###
num_tube_AC_coils = int(np.round((z2_tube - z1_tube) / wavelength)) + 1

# positive current coils
loop_radius_list += [r_tube_AC for i in range(num_tube_AC_coils)]
loop_current_list += [I_tube_AC for i in range(num_tube_AC_coils)]
z0_list += np.ndarray.tolist(np.linspace(z1_tube, z2_tube, num_tube_AC_coils) \
                             + np.mod(-phase, 2 * np.pi) / (2 * np.pi) * wavelength)
color_list += ['r' for i in range(num_tube_AC_coils)]
scale_list += [0.5 for i in range(num_tube_AC_coils)]

# negative current coils
# num_tube_AC_coils -= 1
loop_radius_list += [r_tube_AC for i in range(num_tube_AC_coils)]
loop_current_list += [-I_tube_AC for i in range(num_tube_AC_coils)]
z0_list += np.ndarray.tolist(np.linspace(z1_tube, z2_tube, num_tube_AC_coils) \
                             + np.mod(-phase + np.pi, 2 * np.pi) / (2 * np.pi) * wavelength)
color_list += ['b' for i in range(num_tube_AC_coils)]
scale_list += [0.5 for i in range(num_tube_AC_coils)]

### Plots ###

n_lines = 20
# start_points = (np.array([-r_cube_DC + np.zeros(n_lines),
#                           np.linspace(-0.7*r_cube_DC, 0.7*r_cube_DC, n_lines)]).transpose())
start_points = None

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

    if plot_B_map is True:
        plt.figure(10)
        plot_loop_symbols(z0=z0, loop_radius=loop_radius, scale=scale_list[i],
                          color=color_list[i], length_units='cm')

if plot_B_map is True:
    plt.figure(10)
    plot_magnetic_field_lines(z, r, B_z_tot, B_r_tot,
                              start_points=start_points, length_units='cm')
    plt.xlim([z_calc_B_1 * 1e2, z_calc_B_2 * 1e2])
    plt.ylim([-r_cube_DC * 1.1 * 1e2, r_cube_DC * 1.1 * 1e2])

# plot current as function of position on axis
plt.figure(1)
plt.plot(z0_list, loop_current_list, 'o')
plt.xlabel('z [m]')
plt.ylabel('Current')
plt.title('Current in coils')
# plt.tight_layout()
# plt.legend()
plt.grid(True)

plt.figure(2)
plt.plot(z0_list, loop_radius_list, 'o')
plt.xlabel('z [m]')
plt.ylabel('Coil radius')
plt.title('Radii of coils')
# plt.tight_layout()
# plt.legend()
plt.grid(True)

# plot magnetic field profile on axis
B_z_on_axis = B_z_tot[int(B_z_tot.shape[0] / 2), :]
plt.figure(3)
plt.plot(z_cm, B_z_on_axis,
         label='phase = ' + str(phase / np.pi) + '$\\pi$')
plt.xlabel('z [cm]')
plt.ylabel('$B_z$ [Tesla]')
plt.title('$B_z$ as a function of $z$', size=15)
plt.tight_layout()
plt.legend()
plt.grid(True)

# plot magnetic Bz as a function of r
plt.figure(4)
# z_search_list = [-10, -5, 0, 20, 25, 30]
z_search_list = [-11, 19, 21, 22, 24, 26]
for z_search in z_search_list:
    ind_curr_z = np.where(z_cm > z_search)[0][0]
    curr_z = '{:.2f}'.format(z_cm[ind_curr_z])
    B_z_radial = B_z_tot[:, ind_curr_z]
    # B_z_radial = B_z_tot[int(B_z_tot.shape[1] / 2):, ind_curr_z] # half the r vec
    # label = 'z = ' + str(curr_z) + 'cm, phase = ' + str(phase / np.pi) + '$\\pi$'
    label = 'z = ' + str(curr_z) + 'cm'
    plt.plot(r_cm, B_z_radial, label=label)
plt.xlabel('r [cm]')
plt.ylabel('$B_z$ [Tesla]')
plt.title('$B_z$ as a function of $r$', size=15)
plt.tight_layout()
plt.legend()
plt.grid(True)

# plot magnetic Br as a function of r
plt.figure(5)
z_search_list = [-11, 19, 21, 22, 24, 26]
for z_search in z_search_list:
    ind_curr_z = np.where(z_cm > z_search)[0][0]
    curr_z = '{:.2f}'.format(z_cm[ind_curr_z])
    B_r_radial = B_r_tot[:, ind_curr_z]
    # label = 'z = ' + str(curr_z) + 'cm, phase = ' + str(phase / np.pi) + '$\\pi$'
    label = 'z = ' + str(curr_z) + 'cm'
    plt.plot(r_cm, B_r_radial, label=label)
plt.xlabel('r [cm]')
plt.ylabel('$B_r$ [Tesla]')
plt.title('$B_r$ as a function of $r$', size=15)
plt.tight_layout()
plt.legend()
plt.grid(True)
