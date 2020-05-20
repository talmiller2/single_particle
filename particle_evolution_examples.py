import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from magnetic_field_functions import get_radius, evolve_particle_in_em_fields

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})

plt.close('all')
plot3d_exists = False
# plot3d_exists = True

example_system = 'helix'
# example_system = '2d_static'
# example_system = 'tokamak_banana'
# example_system = 'tokamak_transit'
# example_system = 'magnetic_mirror'

if example_system is 'helix':
    # helix movement in constant magnetic field
    # x_0 = np.array([0, 0, 0])
    x_0 = np.array([1, 0, 0])
    v_0 = np.array([0, -1, 0.5])
    dt = np.pi / 30
    num_steps = 200
    E_function = lambda x, t: 0
    B_function = lambda x, t: np.array([0, 0, 1])

elif example_system is '2d_static':
    # paper example 1: "2D dynamics in a static electromagnetic field"
    x_0 = np.array([0, -1, 0])
    v_0 = np.array([0.1, 0.01, 0])
    dt = np.pi / 10
    num_steps = int(0.7e4)


    def E_function(x, t):
        R = get_radius(x)
        return 1e-2 / R ** 3 * np.array([x[0], x[1], 0])


    def B_function(x, t):
        R = get_radius(x)
        return np.array([0, 0, R])

elif example_system in ['tokamak_banana', 'tokamak_transit']:
    # paper example 2: "2D dynamics in an axisymmetric tokamak geometry"
    x_0 = np.array([1.05, 0, 0])
    if example_system == 'tokamak_banana':
        v_0 = np.array([0, 4.816e-4, -2.059e-3])  # banana orbit
    elif example_system == 'tokamak_transit':
        v_0 = np.array([0, 2 * 4.816e-4, -2.059e-3])  # transit orbit
    # dt = np.pi / 10
    # num_steps = int(5e5)
    # num_steps = int(2e5)
    dt = np.pi / 3
    num_steps = int(1e5)


    def E_function(x, t):
        return np.array([0, 0, 0])


    def B_function(x, t):
        R = get_radius(x)
        B_x = -(2 * x[1] + x[0] * x[2]) / (2 * R ** 2)
        B_y = (2 * x[0] - x[1] * x[2]) / (2 * R ** 2)
        B_z = (R - 1) / (2 * R)
        return np.array([B_x, B_y, B_z])

elif example_system in ['magnetic_mirror']:
    # x_0 = np.array([0.01, -0.02, 0])
    # x_0 = np.array([0.5, -0.5, 0])
    # x_0 = np.array([0.05, -0.05, 0])
    # x_0 = np.array([0, 0, 0])
    # x_0 = np.array([0.05, 0.1, -0.1])
    x_0 = np.array([0.1, 0.2, -0.1])
    # v_hat = np.array([0, 1, 1])
    v_hat = np.array([0, 3, -1])
    # v_hat = np.array([0, -3, -1])
    # v_hat = np.array([0, 10, 1])
    v_magnitude = 0.02
    # v_magnitude = 0.05
    v_0 = v_hat / np.linalg.norm(v_hat) * v_magnitude
    dt = np.pi / 30
    num_steps = int(5e3)


    # dt = np.pi / 100
    # num_steps = int(2e4)

    def E_function(x, t):
        return np.array([0, 0, 0])


    def B_function(x, t):
        R = get_radius(x)
        # return np.array([0, 0, 1 + np.sin(x[2]/10)])
        # return np.array([0, 0, 1 + np.sin(x[2]/10 * 2 * np.pi) + R/10])
        factor = 2 * np.pi
        B_z = 2 + np.sin(factor * x[2])
        B_r = -0.5 * R * factor * np.cos(factor * x[2])  # -0.5*R*dB_z_dz
        B_x = B_r * x[0] / R
        B_y = B_r * x[1] / R
        return np.array([B_x, B_y, B_z])


else:
    raise ValueError('invalid example_system = ' + example_system)

t, x, y, z, vx, vy, vz = evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function, num_steps=num_steps)
R = np.sqrt(x ** 2 + y ** 2)
v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

### Plots
linewidth = 1

plt.figure(1)
plt.plot(t, x, label='x', linewidth=linewidth)
plt.plot(t, y, label='y', linewidth=linewidth)
plt.plot(t, z, label='z', linewidth=linewidth)
plt.legend()
plt.xlabel('t')
plt.ylabel('coordinate')
plt.grid(True)
plt.tight_layout()

plt.figure(2)
plt.plot(x, y, linewidth=linewidth)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()

plt.figure(3)
plt.plot(R, z, label=example_system, linewidth=linewidth)
plt.xlabel('R')
plt.ylabel('z')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(4)
plt.plot(t, vx, label='$v_x$', linewidth=linewidth)
plt.plot(t, vy, label='$v_y$', linewidth=linewidth)
plt.plot(t, vz, label='$v_z$', linewidth=linewidth)
plt.plot(t, v_norm, '--', label='$v_{norm}$', linewidth=linewidth)
plt.legend()
plt.xlabel('t')
plt.ylabel('velocity')
plt.grid(True)
plt.tight_layout()

if plot3d_exists is True:
    plt.figure(5)
else:
    fig = plt.figure(5)
    ax = Axes3D(fig)
ax.plot(x, y, z, label=example_system, linewidth=linewidth)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('particle 3d trajectory')
plt.legend()
