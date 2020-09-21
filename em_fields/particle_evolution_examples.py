import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from em_fields.em_functions import evolve_particle_in_em_fields
from em_fields.magnetic_forms import get_radius

plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})

plt.close('all')
plot3d_exists = False
# plot3d_exists = True

# example_system = 'helix'
# example_system = 'helix_with_RF'
example_system = '2d_static'
# example_system = 'tokamak_banana'
# example_system = 'tokamak_transit'
# example_system = 'magnetic_mirror'

if example_system is 'helix':
    # helix movement in constant magnetic field
    # x_0 = np.array([0, 0, 0])
    x_0 = np.array([1, 0, 0])
    # v_0 = np.array([0, -1, 0.5])
    v_0 = np.array([0, 1, 0.5])
    dt = np.pi / 30
    num_steps = 200
    E_function = lambda x, t: 0
    B_function = lambda x, t: np.array([0, 0, 1])

if example_system is 'helix_with_RF':
    # helix movement in constant magnetic field
    # x_0 = np.array([0, 0, 0])
    x_0 = np.array([1, 0, 0])
    # v_0 = np.array([0, -1, 0.5])
    v_0 = np.array([0, 1, 0.5])
    # v_0 = np.array([0, 1, -0.5])
    # v_0 = np.array([0, 1, 0])
    # v_0 = np.array([0, 0, 0])
    dt = np.pi / 30
    num_steps = 1000
    m = 1.0
    e = 1.0

    # B0 = 0
    B_z = 1.0
    # B0 = -1.0

    if B_z == 0:  # pick a default
        anticlockwise = 1
    else:
        anticlockwise = np.sign(B_z)

    # E_RF = 0.0
    # E_RF = 0.01
    # E_RF = 0.1
    E_RF = 0.2
    # E_RF = -0.2
    # E_RF = -0.5
    # E_RF = 1.0

    phase_RF = 0
    # phase_RF = np.pi / 4
    # phase_RF = np.pi / 2
    # phase_RF = np.pi

    omega_cyclotron = e * np.abs(B_z) / m
    if omega_cyclotron == 0:  # pick a default
        omega = 1
    else:
        omega = omega_cyclotron  # resonance
    # omega *= 1.1 # off resonance
    # omega *= 0.9 # off resonance
    # omega *= 0.5 # off resonance
    # omega *= 2.0 # off resonance
    # omega *= 0.1 # off resonance

    # c = np.inf
    # c = 3e8
    c = 100
    # c = 10

    k = omega / c


    # k = - omega / c
    # k = 0

    def E_function(x, t):
        z = x[2]
        return E_RF * np.array([anticlockwise * np.sin(k * z - omega * t + phase_RF),
                                np.cos(k * z - omega * t + phase_RF),
                                0])


    def B_function(x, t):
        B_axial = np.array([0, 0, B_z])
        # B_RF = 1/c * k_hat cross E_RF
        # https://en.wikipedia.org/wiki/Sinusoidal_plane-wave_solutions_of_the_electromagnetic_wave_equation
        # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
        z = x[2]
        B_RF = E_RF / c * np.array([-np.sign(k) * np.cos(k * z - omega * t + phase_RF),
                                    np.sign(k) * anticlockwise * np.sin(k * z - omega * t + phase_RF),
                                    0])
        return B_axial + B_RF

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


    def E_function(x, t):
        return np.array([0, 0, 0])


else:
    raise ValueError('invalid example_system = ' + example_system)

hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function, num_steps=num_steps)
t = hist['t']
x = hist['x'][:, 0]
y = hist['x'][:, 1]
z = hist['x'][:, 2]
vx = hist['v'][:, 0]
vy = hist['v'][:, 1]
vz = hist['v'][:, 2]

R = np.sqrt(x ** 2 + y ** 2)
v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

### Plots
linewidth = 2

plt.figure(1)
# plt.subplot(1,2,1)
plt.plot(t, x, label='x', linewidth=linewidth, color='b')
plt.plot(t, y, label='y', linewidth=linewidth, color='g')
plt.plot(t, z, label='z', linewidth=linewidth, color='r')
plt.plot(t, R, label='R', linewidth=linewidth, color='k')
plt.legend()
plt.xlabel('t')
plt.ylabel('coordinate')
plt.grid(True)
plt.tight_layout()

plt.figure(4)
# plt.subplot(1,2,2)
plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
plt.plot(t, vz, label='$v_z$', linewidth=linewidth, color='r')
plt.plot(t, v_norm, label='$v_{norm}$', linewidth=linewidth, color='k')
plt.legend()
plt.xlabel('t')
plt.ylabel('velocity')
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

plt.figure(5)
E = 0.5 * v_norm ** 2.0
# plt.plot(t, E / E[0], label='$E/E_0$', linewidth=linewidth)
plt.plot(t, E, label='$E$', linewidth=linewidth)
plt.legend()
plt.xlabel('t')
plt.ylabel('kinetic energy')
plt.grid(True)
plt.tight_layout()

if plot3d_exists is True:
    plt.figure(6)
else:
    fig = plt.figure(6)
    ax = Axes3D(fig)
ax.plot(x, y, z, label=example_system, linewidth=linewidth)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('particle 3d trajectory')
plt.legend()
