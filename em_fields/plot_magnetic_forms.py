import matplotlib.pyplot as plt
import numpy as np

from em_fields.magnetic_forms import magnetic_field_logan, magnetic_field_jaeger, magnetic_field_post

plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})
# plt.rcParams.update({'lines.linestyle': ':'})

plt.close('all')

l = 1
# z = np.linspace(0, l, 1000)
z = np.linspace(-l, l, 1000)

# mirror ratio
Rm = 3
# Rm = 7
# Rm = 10

x = np.array([z, z, z])

B = magnetic_field_logan(x, 1, Rm, l)
B_logan = B[2]

B = magnetic_field_jaeger(x, 1, Rm, l)
B_jaeger = B[2]

B = magnetic_field_post(x, 1, Rm, l)
B_post = B[2]

# plot axial magnetic fields
plt.figure(1)
plt.plot(z, B_logan, label='Logan et al', color='b')
plt.plot(z, B_post, label='Post', color='r')
plt.plot(z, B_jaeger, label='Jaeger et al', color='g')
plt.legend()
plt.xlabel('z [m]')
plt.ylabel('B [T]')
plt.grid(True)
plt.tight_layout()
