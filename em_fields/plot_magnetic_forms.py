import matplotlib.pyplot as plt
import numpy as np

from em_fields.magnetic_forms import magnetic_field_logan, magnetic_field_jaeger, magnetic_field_post

plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})
# plt.rcParams.update({'lines.linestyle': ':'})

plt.close('all')

l = 1
# l = 5
# z = np.linspace(0, l, 1000)
z = np.linspace(-l, l, 1000)
# z = np.linspace(-2*l, 2*l, 1000)

# mirror ratio
Rm = 3
# Rm = 7
# Rm = 10

# x = np.array([0*z, 0*z, z])
x = np.array([0 * z + l / 2, 0 * z + l / 2, z])

field_dict = {'z_mirror_shift': 0, 'B0': 1, 'Rm': Rm, 'l': 1}
# field_dict = {'z_mirror_shift': l / 2, 'B0': 1, 'Rm': Rm, 'l': l}

B, _ = magnetic_field_logan(x, field_dict)
B_logan = B

B, _ = magnetic_field_jaeger(x, field_dict)
B_jaeger = B

B, _ = magnetic_field_post(x, field_dict)
B_post = B

B_s = 1.0
sigma_s = l / 5.0
z_mod = np.mod(z, l)
B_slope = B_s * (z_mod - l / 2.0) / l \
          * (1 - np.exp(- (z_mod - l) ** 2.0 / sigma_s ** 2.0)) \
          * (1 - np.exp(- (z_mod) ** 2.0 / sigma_s ** 2.0))

B_logan_slope = B_logan[2] + B_slope
B_post_slope = B_post[2] + B_slope

z /= l

# plot axial magnetic fields
plt.figure(1, figsize=(8, 4))
# plt.plot(z, B_logan[2], label='Logan', color='r')
# plt.plot(z, B_jaeger[2], '--', label='Jaeger et al', color='g')
# plt.plot(z, B_post[2], label='Post', color='r')
plt.plot(z, B_post, label='Post', color='b')
# plt.plot(z, B_slope, '--', label='slope', color='k')
# plt.plot(z, B_logan_slope, '--', label='Logan + slope', color='b')
# plt.plot(z, B_post_slope, '--', label='Post + slope', color='r')

### plot shallow regions of magnetic field near the center
# z_bounds_list = [[-1, 0], [0, 1]]
# for z_bounds in z_bounds_list:
#     B_array = B_post[2]
#     B_max_reversal = 1.25
#     inds = [i for i, Bi in enumerate(B_array) if Bi <= B_max_reversal and z[i] >= z_bounds[0] and z[i] <= z_bounds[1]]
#     z_array = z[inds]
#     B_lower_bound = B_array[inds]
#     B_upper_bound = B_max_reversal + B_lower_bound * 0
#     plt.fill_between(z_array, B_lower_bound, B_upper_bound, color='b', alpha=0.3)

# plt.legend()
# plt.x_label('z [m]')
plt.xlabel('$z / l$')
plt.ylabel('$B_z$ [T]')
plt.grid(True)
plt.tight_layout()
# plt.layout_engine()

# # plot radial magnetic field
# plt.figure(2)
# plt.plot(z, B_logan[0], label='Logan et al', color='b')
# # plt.plot(z, B_jaeger[0], '--', label='Jaeger et al', color='g')
# plt.plot(z, B_post[0], label='Post', color='r')
# plt.legend()
# plt.x_label('z [m]')
# plt.y_label('$B_x$ [T]')
# plt.grid(True)
# plt.tight_layout()

# plt.figure(3)
# plt.plot(z, B_logan[1], label='Logan et al', color='b')
# # plt.plot(z, B_jaeger[1], '--', label='Jaeger et al', color='g')
# plt.plot(z, B_post[1], label='Post', color='r')
# plt.legend()
# plt.x_label('z [m]')
# plt.y_label('$B_y$ [T]')
# plt.grid(True)
# plt.tight_layout()

## save plots to file
# save_dir = '../../../Papers/texts/paper2022/pics/'
# file_name = 'axial_magnetic_field_post_form'
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
# # beingsaved.savefig(save_dir + file_name + '.jpeg', format='jpeg', dpi=300)
