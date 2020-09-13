import matplotlib.pyplot as plt
import numpy as np

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

# magnetic fields from Logan et al, 1972
gamma = Rm - 1
lamda = 5.5
B_logan = 1 / Rm * (1 + gamma * np.sin(np.pi * z / l) ** 2.0)
B_post = 1 / Rm * (1 + gamma * np.exp(- lamda * np.sin(np.pi * (z - l / 2) / l) ** 2.0))

# magnetic field from Jaeger et al, 1972
# B_jaeger = 1 / Rm * (2 - np.cos(2 * np.pi * z / l))
# modified to make more sense for general Rm
B_jaeger = 1 / Rm * (1 + (Rm - 1) / 2.0 * (1 - np.cos(2 * np.pi * z / l)))

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
