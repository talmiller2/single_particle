import matplotlib.pyplot as plt
import numpy as np

from loop_fields.loop_functions import get_current_loop_magnetic_field_cartesian
from loop_fields.plot_functions import plot_loop_symbols, plot_magnetic_field_lines

plt.close('all')

# plot loop field test
dl = 0.05
# y = np.arange(-0.5, 0.5, dl)
# z = np.arange(-0.5, 0.5, dl)
y = np.arange(-2, 2, dl)
z = np.arange(-2, 2, dl)
# y = np.arange(-1.5, 1.5, dl)
# z = np.arange(-3, 3, dl)
# y = np.arange(-4, 4, dl)
# z = np.arange(-3, 3, dl)
# dl = 0.3
# y = np.arange(-5, 5, dl)
# z = np.arange(-5, 5, dl)
# dl = 0.2
# y = np.arange(-15, 15, dl)
# z = np.arange(-15, 15, dl)
# print('y shape = ' + str(y.shape))
# print('z shape = ' + str(z.shape))
# yy, zz = np.meshgrid(y, z, sparse=False)
yy, zz = np.meshgrid(y, z, indexing='ij')
print('yy shape = ' + str(yy.shape))
print('zz shape = ' + str(zz.shape))
# xx = 0*yy + 0.1
# xx = 0*yy + 1e-6
xx = 0 * yy + 0

n_y = len(y)
n_z = len(z)
points = np.empty([n_y * n_z, 3])
for i in range(0, n_y):
    for j in range(0, n_z):
        points[n_z * i + j, :] = np.array([0, y[i], z[j]])

### single loop
loop_radius_list = [1.0]
loop_current_list = [1.0]
z0_list = [0]

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
# loop_radius_list = [1.0, np.sqrt(4.0/7.0), np.sqrt(4.0/7.0)]
# loop_current_list = [1.0, 1.0, 1.0]
# z0_list = [0, -np.sqrt(3.0/7.0), np.sqrt(3.0/7.0)]

### other double coil
# loop_radius_list = [1.0, 1.0]
# loop_current_list = [1.0, 1.0]
# z0_list = [-1.5, 0.5]

### Magnetic mirror v1
# # loop_radius_list = [1.0, 1.0, 1.0]
# # loop_current_list = [5.0, 1.0, 5.0]
# z0_list = [-1.0, 0.0, 1.0]

### Magnetic mirror v2
# loop_radius_list = [0.5, 1.0, 0.5]
# loop_current_list = [1.0, 1.0, 1.0]
# z0_list = [-1.0, 0.0, 1.0]

### Magnetic mirror v3
# loop_radius_list = [0.5, 0.8, 1.0, 0.8, 0.5]
# loop_current_list = [2.0, 1.5, 1.0, 1.5, 2.0]
# z0_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

### Multi mirror
# loop_radius_list = [1.0, 0.5, 1.0, 0.5, 1.0]
# loop_current_list = [1.0, 1.0, 1.0, 1.0, 1.0]
# z0_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

B_x_tot, B_y_tot, B_z_tot = 0, 0, 0

for i in range(len(loop_radius_list)):
    loop_radius = loop_radius_list[i]
    loop_current = loop_current_list[i]
    x0 = 0
    y0 = 0
    # y0 = 1.0
    z0 = z0_list[i]

    B_x, B_y, B_z = get_current_loop_magnetic_field_cartesian(xx, yy, zz, x0=x0, y0=y0, z0=z0, loop_radius=loop_radius)

    B_x_tot += B_x
    B_y_tot += B_y
    B_z_tot += B_z

    plt.figure(6)
    # scale = 1.0
    scale = loop_current / np.mean(loop_current_list)
    plot_loop_symbols(z0=z0, loop_radius=loop_radius, scale=scale)

plt.figure(6)
plot_magnetic_field_lines(z, y, B_z_tot, B_y_tot)
