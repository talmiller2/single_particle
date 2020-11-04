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

# l = 1
l = 5
# z = np.linspace(0, l, 1000)
z = np.linspace(-l, l, 1000)
# z = np.linspace(-2*l, 2*l, 1000)

# mirror ratio
Rm = 3
# Rm = 7
# Rm = 10

# x = np.array([0*z, 0*z, z])
x = np.array([0 * z + l / 2, 0 * z + l / 2, z])

B = magnetic_field_logan(x, 1, Rm, l)
B_logan = B

B = magnetic_field_jaeger(x, 1, Rm, l)
B_jaeger = B

B = magnetic_field_post(x, 1, Rm, l)
B_post = B

# plot axial magnetic fields
plt.figure(1)
plt.plot(z, B_logan[2], label='Logan et al', color='b')
# plt.plot(z, B_jaeger[2], '--', label='Jaeger et al', color='g')
plt.plot(z, B_post[2], label='Post', color='r')
plt.legend()
plt.xlabel('z [m]')
plt.ylabel('$B_z$ [T]')
plt.grid(True)
plt.tight_layout()

# plot radial magnetic field
plt.figure(2)
plt.plot(z, B_logan[0], label='Logan et al', color='b')
# plt.plot(z, B_jaeger[0], '--', label='Jaeger et al', color='g')
plt.plot(z, B_post[0], label='Post', color='r')
plt.legend()
plt.xlabel('z [m]')
plt.ylabel('$B_x$ [T]')
plt.grid(True)
plt.tight_layout()

plt.figure(3)
plt.plot(z, B_logan[1], label='Logan et al', color='b')
# plt.plot(z, B_jaeger[1], '--', label='Jaeger et al', color='g')
plt.plot(z, B_post[1], label='Post', color='r')
plt.legend()
plt.xlabel('z [m]')
plt.ylabel('$B_y$ [T]')
plt.grid(True)
plt.tight_layout()
