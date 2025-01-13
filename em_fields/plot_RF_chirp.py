import matplotlib.pyplot as plt
import numpy as np

from em_fields.default_settings import define_default_settings
from em_fields.magnetic_forms import magnetic_field_logan, magnetic_field_post

plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})
# plt.rcParams.update({'lines.linestyle': ':'})

plt.close('all')

settings = define_default_settings()

l = 1
# l = 5
z = np.linspace(0, l, 1000)
# z = np.linspace(-l, l, 1000)
# z = np.linspace(l / 2, l, 1000)
# z = np.linspace(-2*l, 2*l, 1000)

# mirror ratio
Rm = 3
# Rm = 7
# Rm = 10

# x = np.array([0*z, 0*z, z])
x = np.array([0 * z + l / 2, 0 * z + l / 2, z])
field_dict = {'B0': 1, 'Rm': Rm, 'l': l, 'z0': 0, 'use_transverse_fields': True}
B_logan = magnetic_field_logan(x, field_dict)[2, :]
B_post = magnetic_field_post(x, field_dict)[2, :]

B_s = 0.2
# B_s = 1.0
# B_s = 0.0
sigma_s = l / 5.0
z_mod = np.mod(z, l)
B_slope = B_s * (z_mod - l / 2.0) / l \
          * (1 - np.exp(- (z_mod - l) ** 2.0 / sigma_s ** 2.0)) \
          * (1 - np.exp(- (z_mod) ** 2.0 / sigma_s ** 2.0))

B_logan_slope = B_logan + B_slope
B_post_slope = B_post + B_slope

v_z_0 = 1.0 * settings['v_th']
# v_t_0 = 1.0 * v_z_0
v_t_0 = 0.1 * v_z_0

dz = z[1] - z[0]

v_z_logan = np.sqrt(v_z_0 ** 2 + v_t_0 ** 2 * (1 - (B_logan) ** 2))
t_logan = np.cumsum(dz / v_z_logan)

v_z_logan_slope = np.sqrt(v_z_0 ** 2 + v_t_0 ** 2 * (1 - (B_logan_slope) ** 2))
t_logan_slope = np.cumsum(dz / v_z_logan_slope)

v_z_post = np.sqrt(v_z_0 ** 2 + v_t_0 ** 2 * (1 - (B_post) ** 2))
t_post = np.cumsum(dz / v_z_post)

v_z_post_slope = np.sqrt(v_z_0 ** 2 + v_t_0 ** 2 * (1 - (B_post_slope) ** 2))
t_post_slope = np.cumsum(dz / v_z_post_slope)

z /= l

# plot axial magnetic fields
plt.figure(1)
# plt.plot(z, B_logan, label='Logan', color='b')
plt.plot(z, B_post, label='Post', color='r')
# plt.plot(z, B_slope, '--', label='slope', color='k')
# plt.plot(z, B_logan_slope, '--', label='Logan + slope', color='b')
# plt.plot(z, B_post_slope, '--', label='Post + slope', color='r')
plt.plot(z, B_post_slope, '-', label='Post + slope', color='b')
plt.legend()
# plt.x_label('z [m]')
plt.xlabel('z / l')
plt.ylabel('$B_z$ [T]')
plt.grid(True)
plt.tight_layout()

plt.figure(2)
# plt.plot(t_logan, v_z_logan, label='Logan', color='b')
# plt.plot(t_logan_slope, v_z_logan_slope, '--', label='Logan + slope', color='b')
plt.plot(t_post, v_z_post, label='Post', color='r')
plt.plot(t_post_slope, v_z_post_slope, '--', label='Post + slope', color='r')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('$v_z$ [m/s]')
plt.grid(True)
plt.tight_layout()

plt.figure(3)
# plt.plot(t_logan, B_logan, label='Logan', color='b')
# plt.plot(t_logan_slope, B_logan_slope, '--', label='Logan + slope', color='b')
plt.plot(t_post, B_post, label='Post', color='r')
plt.plot(t_post_slope, B_post_slope, '--', label='Post + slope', color='r')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('$\\omega(t) \propto B_z$')
plt.grid(True)
plt.tight_layout()

### saving figure
save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/research_update/pics/'

file_name = 'mirror_slope_shape'
beingsaved = plt.figure(1)
beingsaved.savefig(save_dir + file_name + '.pdf', format='pdf')
