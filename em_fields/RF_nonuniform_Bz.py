import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from em_fields.default_settings import define_default_settings
from em_fields.em_functions import evolve_particle_in_em_fields, get_thermal_velocity, get_cyclotron_angular_frequency
from em_fields.magnetic_forms import magnetic_field_post

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})

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

Rm = 3.0
print('Rm = ' + str(Rm))

loss_cone_angle = np.arcsin(Rm ** (-0.5)) * 360 / (2 * np.pi)
print('loss_cone_angle = ' + str(loss_cone_angle))

# c = np.inf
c = 3e8
# c = 3e7
# c = 3e6
# c = 100
# c = 10

print('v_th / c = ', v_th / c)

# B0 = 0 # Tesla
# B0 = 0.1  # Tesla
# B0 = 0.2  # Tesla
B0 = 1.0  # Tesla
# B0 = -1.0 # Tesla

omega_cyclotron = get_cyclotron_angular_frequency(q, B0, m)
# omega_cyclotron = get_cyclotron_angular_frequency(q, B0 * Rm, m)
# omega_cyclotron = get_cyclotron_angular_frequency(q, B0 * Rm * 0.5, m)
tau_cyclotron = 2 * np.pi / omega_cyclotron

angle_to_z_axis = 0  # deg
# angle_to_z_axis = 0.5 * loss_cone_angle # deg
# angle_to_z_axis = 0.95 * loss_cone_angle # deg
# angle_to_z_axis = 0.99 * loss_cone_angle # deg
# angle_to_z_axis = loss_cone_angle # deg
# angle_to_z_axis = 1.01 * loss_cone_angle # deg
# angle_to_z_axis = 1.10 * loss_cone_angle # deg
print('angle_to_z_axis = ' + str(angle_to_z_axis) + ' degrees')

angle_to_z_axis_rad = angle_to_z_axis / 360 * 2 * np.pi
v_0 = np.array([0, np.sin(angle_to_z_axis_rad), np.cos(angle_to_z_axis_rad)])

# v_0 = np.array([0, 1, 10])
# v_0 = np.array([0, 10, 1])
# v_0 = np.array([0, 2, 1])
# v_0 = np.array([0, 1, 1])
# v_0 = np.array([0, 0.5, 1])

# normalize the velocity vector
v_0 = v_0 / np.linalg.norm(v_0)
v_0 *= v_th
v_0_norm = np.linalg.norm(v_0)

cyclotron_radius = np.linalg.norm(v_0) / omega_cyclotron
# x_0 = np.array([0, 0, 0])
# x_0 = cyclotron_radius * np.array([1, 0, 0])
# x_0 = cyclotron_radius * np.array([0, 0, 0])
# x_0 = np.array([0, 0, 0])
# x_0 = np.array([0, 0, 0.5])
x_0 = np.array([0, 0, 0])

l = 1.0  # m (interaction length)
# l = 2.0  # m (interaction length)
# l = 100.0  # m (interaction length)
v_z = v_0[2]
# t_max = l / v_z
# t_max = 2 * l / v_z
t_max = 10 * l / v_z
# t_max = min(t_max, 100 * tau_cyclotron)

# dt = tau_cyclotron / 50 / Rm
# dt = tau_cyclotron / 200
dt = tau_cyclotron / 20
# num_steps = 1000
num_steps = int(t_max / dt)
# num_steps = min(num_steps, 3000)

print('num_steps = ', num_steps)
print('t_max = ', num_steps * dt, 's')

if B0 == 0:  # pick a default
    anticlockwise = 1
else:
    anticlockwise = np.sign(B0)

E_RF = 0.0
# E_RF = 1  # kV/m
# E_RF = 3  # kV/m
# E_RF = 5  # kV/m
# E_RF = 10  # kV/m
# E_RF = 15  # kV/m
# E_RF = 20  # kV/m
# E_RF = 50  # kV/m
# E_RF = 75  # kV/m
# E_RF = 80  # kV/m
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
    use_transverse_fields = True
    # use_transverse_fields = False

    # B_mirror = magnetic_field_constant(B0)
    # B_mirror = magnetic_field_logan(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)
    # B_mirror = magnetic_field_jaeger(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)
    B_mirror = magnetic_field_post(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)

    # B_RF = 1/c * k_hat cross E_RF
    # https://en.wikipedia.org/wiki/Sinusoidal_plane-wave_solutions_of_the_electromagnetic_wave_equation
    # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
    z = x[2]
    B_RF = E_RF / c * np.array([-np.sign(k) * np.cos(k * z - omega * t + phase_RF),
                                np.sign(k) * anticlockwise * np.sin(k * z - omega * t + phase_RF),
                                0])
    # B_RF = 0 # test that does not satisfy Maxwell equations.
    return B_mirror + B_RF


hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function,
                                    num_steps=num_steps, q=q, m=m, return_fields=True)
t = hist['t']
x = hist['x'][:, 0]
y = hist['x'][:, 1]
z = hist['x'][:, 2]
vx = hist['v'][:, 0]
vy = hist['v'][:, 1]
vz = hist['v'][:, 2]
E = hist['E']
B = hist['B']

R = np.sqrt(x ** 2 + y ** 2)
v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

# scale all observables
t /= tau_cyclotron
x /= cyclotron_radius
y /= cyclotron_radius
# z /= cyclotron_radius
R /= cyclotron_radius
z /= l
# R /= l
vx /= v_th
vy /= v_th
vz /= v_th
v_norm /= v_th

### Plots
linewidth = 2

plt.figure(1)
# plt.subplot(1,2,1)
# plt.plot(t, x, label='x', linewidth=linewidth, color='b')
# plt.plot(t, y, label='y', linewidth=linewidth, color='g')
plt.plot(t, R, label='$R/r_{cyc}$', linewidth=linewidth, color='b')
plt.plot(t, z, label='$z/l$', linewidth=linewidth, color='r')
plt.legend()
plt.xlabel('t / $\\tau_{cyc}$')
# plt.ylabel('coordinate / $r_{cyc}$')
plt.ylabel('coordinate (normalized)')
plt.grid(True)
plt.tight_layout()

plt.figure(4)
# plt.subplot(1,2,2)
# plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
# plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
v_r = np.sqrt(vx ** 2 + vy ** 2)
plt.plot(t, v_r, label='$v_r$', linewidth=linewidth, color='b')
v_perp_loss_cone = v_norm * Rm ** (-0.5)
plt.plot(t, v_perp_loss_cone, label='$v_{r,LC}$', linewidth=linewidth, linestyle='--', color='c')
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
# plt.xlabel('R / $r_{cyc}$')
# plt.ylabel('z / $r_{cyc}$')
plt.xlabel('R (normalized)')
plt.ylabel('z (normalized)')
plt.grid(True)
# plt.legend()
plt.tight_layout()

plt.figure(5)
E_kin = 0.5 * v_norm ** 2.0
# plt.plot(t, E / E[0], linewidth=linewidth)
plt.plot(t, (E_kin - E_kin[0]) / E_kin[0] * 100, linewidth=linewidth)
plt.xlabel('t / $\\tau_{cyc}$')
# plt.ylabel('$E / E_0$')
plt.ylabel('kinetic energy % increase')
plt.grid(True)
plt.tight_layout()
print('energy change: ' + str((E_kin[-1] - E_kin[0]) / E_kin[0] * 100) + '%')

plt.figure(6)
# plt.plot(t, B[:, 0], label='$B_x$', linewidth=linewidth, color='b')
# plt.plot(t, B[:, 1], label='$B_y$', linewidth=linewidth, color='g')
plt.plot(t, B[:, 0], label='$B_r$', linewidth=linewidth, color='b')
plt.plot(t, B[:, 2], label='$B_z$', linewidth=linewidth, color='r')
plt.xlabel('t / $\\tau_{cyc}$')
plt.ylabel('Tesla')
plt.grid(True)
plt.tight_layout()
plt.legend()

if plot3d_exists is True:
    plt.figure(7)
else:
    fig = plt.figure(7)
    ax = Axes3D(fig)
ax.plot(x, y, z, linewidth=linewidth)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
ax.set_xlabel('x (normalized)')
ax.set_ylabel('y (normalized)')
ax.set_zlabel('z (normalized)')
ax.set_title('particle 3d trajectory')
# plt.legend()
