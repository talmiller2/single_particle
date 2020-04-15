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
# points = np.empty([n_y * n_z, 3])
# for i in range(0, n_y):
#   for j in range(0, n_z):
#     points[n_z * i + j, :] = np.array([0, y[i], z[j]])

# n_lines = math.floor(n_z / 2)
n_lines = 20
max_r = np.max(r)
min_r = np.min(r)
# # max_y = 0.9
# # min_y = -0.9

# start_points = None
start_points = (np.array([np.zeros(n_lines),
                          min_r + (max_r - min_r) *
                          (0.5 + np.linspace(0, n_lines - 1, n_lines)) /
                          (n_lines)]).transpose()
                )

### single loop
# loop_radius_list = [1.0]
# loop_current_list = [1.0]
# z0_list = [0]

### single loop shifted
# loop_radius_list = [1.0]
# loop_current_list = [1.0]
# z0_list = [-1.0]

### Helmholtz coil
# loop_radius_list = [1.0, 1.0]
# loop_current_list = [1.0, 1.0]
# z0_list = [-0.5, 0.5]
# loop_radius_list = [10.0, 10.0]
# loop_current_list = [1.0, 1.0]
# z0_list = [-5, 5]

### Maxwell coil
loop_radius_list = [1.0, np.sqrt(4.0 / 7.0), np.sqrt(4.0 / 7.0)]
loop_current_list = [1.0, 1.0, 1.0]
z0_list = [0, -np.sqrt(3.0 / 7.0), np.sqrt(3.0 / 7.0)]

### other double coil
# loop_radius_list = [1.0, 1.0]
# loop_current_list = [1.0, 1.0]
# z0_list = [-1.5, 0.5]

### Magnetic mirror v1
# loop_radius_list = [1.0, 1.0, 1.0]
# loop_current_list = [5.0, 1.0, 5.0]
# z0_list = [-1.0, 0.0, 1.0]

### Magnetic mirror v2
# loop_radius_list = [0.5, 1.0, 0.5]
# loop_current_list = [1.0, 1.0, 1.0]
# z0_list = [-1.0, 0.0, 1.0]

### Magnetic mirror v3
# loop_radius_list = [0.5, 0.8, 1.0, 0.8, 0.5]
# loop_current_list = [2.0, 1.5, 1.0, 1.5, 2.0]
# z0_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

### Magnetic mirror v4
# loop_radius_list = [1, 1]
# loop_current_list = [1, 1]
# z0_list = [-5, 5]

### Multi mirror v1
# loop_radius_list = [1, 1, 1, 1]
# loop_current_list = [1, 1, 1, 1]
# z0_list = [-15, -5, 5, 15]

### Main section + MM
# loop_radius_list = [1, 1, 1, 1, 1] + [1, 1, 1, 1]
# # loop_current_list = [1, 1, 1, 1, 1] + [0.1, 1, 0.1, 1]
# loop_current_list = [1, 1, 1, 1, 1] + [1, 0.1, 1, 0.1]
# z0_list = [-15, -12.5, -10, -7.5, -5] + [0, 5, 10, 15]

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

    plt.figure(6)
    # scale = 1.0
    scale = 0.5 * loop_current
    # scale = loop_current / np.mean(loop_current_list)
    plot_loop_symbols(z0=z0, loop_radius=loop_radius, scale=scale)

plt.figure(6)
plot_magnetic_field_lines(z, r, B_z_tot, B_r_tot, start_points=start_points)
