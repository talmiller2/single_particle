import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from em_fields.default_settings import define_default_settings
from em_fields.em_functions import evolve_particle_in_em_fields, get_thermal_velocity, get_cyclotron_angular_frequency

plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})
# plt.rcParams.update({'lines.linestyle': ':'})
#
plt.close('all')

plot3d_exists = False
# plot3d_exists = True

settings = define_default_settings()
m = settings['mi']
q = settings['Z_ion'] * settings['e']  # Coulomb
T_eV = 1e3
# T_eV = 10000e3
kB_eV = settings['kB_eV']
v_th = get_thermal_velocity(T_eV, m, kB_eV)

# c = np.inf
c = 3e8
# c = 3e7
# c = 3e6
# c = 100
# c = 10

print('v_th / c = ', v_th / c)

# B_z = 0 # Tesla
B_z = 1.0  # Tesla
# B_z = -1.0 # Tesla

omega_cyclotron = get_cyclotron_angular_frequency(q, B_z, m)
tau_cyclotron = 2 * np.pi / omega_cyclotron

# v_0 = v_th * np.array([0, -1, 0.5])
v_0 = np.array([0, 1, 10])
# v_0 = np.array([0, 10, 1])
# v_0 = np.array([0, 1, 1])
# v_0 /= np.linalg.norm(v_0)
v_0 = v_0 / np.linalg.norm(v_0)
v_0 *= v_th

cyclotron_radius = np.linalg.norm(v_0) / omega_cyclotron
# x_0 = np.array([0, 0, 0])
x_0 = cyclotron_radius * np.array([1, 0, 0])

l = 1.0  # m (interaction length)
# l = 100.0  # m (interaction length)
v_z = v_0[2]
t_max = l / v_z
# t_max = min(t_max, 100 * tau_cyclotron)

dt = tau_cyclotron / 50
# dt = tau_cyclotron / 200
# num_steps = 1000
num_steps = int(t_max / dt)
# num_steps = min(num_steps, 3000)

print('num_steps = ', num_steps)
print('t_max = ', num_steps * dt, 's')

if B_z == 0:  # pick a default
    anticlockwise = 1
else:
    anticlockwise = np.sign(B_z)

# E_RF = 0.0
E_RF = 1  # kV/m
# E_RF = 10  # kV/m
# E_RF = 100  # kV/m
E_RF *= 1e3  # the SI units is V/m

phase_RF = 0
# phase_RF = np.pi / 4
# phase_RF = np.pi / 2
# phase_RF = np.pi
# phase_RF = 1.5 * np.pi

omega = omega_cyclotron  # resonance
# omega *= 1.01 # off resonance
# omega *= 1.02 # off resonance
# omega *= 1.05 # off resonance
# omega *= 1.1 # off resonance
# omega *= 0.9 # off resonance
# omega *= 0.5 # off resonance
# omega *= 2.0 # off resonance
# omega *= 0.1 # off resonance

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
    # B_RF = 0 # test that does not satisfy Maxwell equations.
    return B_axial + B_RF


t, x, y, z, vx, vy, vz = evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function,
                                                      num_steps=num_steps, q=q, m=m)
R = np.sqrt(x ** 2 + y ** 2)
v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

# scale all observables
t /= tau_cyclotron
x /= cyclotron_radius
y /= cyclotron_radius
z /= cyclotron_radius
R /= cyclotron_radius
vx /= v_th
vy /= v_th
vz /= v_th
v_norm /= v_th

### Plots
linewidth = 2

plt.figure(1)
# plt.subplot(1,2,1)
plt.plot(t, x, label='x', linewidth=linewidth, color='b')
plt.plot(t, y, label='y', linewidth=linewidth, color='g')
plt.plot(t, z, label='z', linewidth=linewidth, color='r')
plt.plot(t, R, label='R', linewidth=linewidth, color='k')
plt.legend()
plt.xlabel('t / $\\tau_{cyc}$')
plt.ylabel('coordinate / $r_{cyc}$')
plt.grid(True)
plt.tight_layout()

plt.figure(4)
# plt.subplot(1,2,2)
plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
plt.plot(t, vz, label='$v_z$', linewidth=linewidth, color='r')
plt.plot(t, v_norm, label='$v_{norm}$', linewidth=linewidth, color='k')
plt.legend()
plt.xlabel('t / $\\tau_{cyc}$')
plt.ylabel('velocity / $v_{th}$')
plt.grid(True)
plt.tight_layout()

plt.figure(2)
plt.plot(x, y, linewidth=linewidth)
plt.xlabel('x / $r_{cyc}$')
plt.ylabel('y / $r_{cyc}$')
plt.grid(True)
plt.tight_layout()

plt.figure(3)
plt.plot(R, z, linewidth=linewidth)
plt.xlabel('R / $r_{cyc}$')
plt.ylabel('z / $r_{cyc}$')
plt.grid(True)
# plt.legend()
plt.tight_layout()

plt.figure(5)
E = 0.5 * v_norm ** 2.0
# plt.plot(t, E / E[0], linewidth=linewidth)
plt.plot(t, (E - E[0]) / E[0] * 100, linewidth=linewidth)
plt.xlabel('t / $\\tau_{cyc}$')
# plt.ylabel('$E / E_0$')
plt.ylabel('kinetic energy % increase')
plt.grid(True)
plt.tight_layout()

print('energy change: ' + str((E[-1] - E[0]) / E[0] * 100) + '%')

if plot3d_exists is True:
    plt.figure(6)
else:
    fig = plt.figure(6)
    ax = Axes3D(fig)
ax.plot(x, y, z, linewidth=linewidth)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('particle 3d trajectory')
# plt.legend()
