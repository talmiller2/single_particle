import matplotlib.pyplot as plt

from em_fields.default_settings import define_default_settings
from em_fields.em_functions import evolve_particle_in_em_fields, get_thermal_velocity, get_cyclotron_angular_frequency
from em_fields.magnetic_forms import *

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})

# plt.close('all')

plot3d_exists = False
# plot3d_exists = True

settings = define_default_settings()
m = settings['mi']
q = settings['Z_ion'] * settings['e']  # Coulomb
# T_eV = 1e3
T_eV = 3e3
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

# print('v_th / c = ', v_th / c)

# B0 = 0 # Tesla
# B0 = 0.01  # Tesla
# B0 = 0.1  # Tesla
# B0 = 0.2  # Tesla
# B0 = 0.5  # Tesla
B0 = 1.0  # Tesla
# B0 = -1.0 # Tesla
# B0 = 5.0  # Tesla

omega_cyclotron = get_cyclotron_angular_frequency(q, B0, m)
tau_cyclotron = 2 * np.pi / omega_cyclotron

angle_to_z_axis = 0  # deg
# angle_to_z_axis = 0.1 * loss_cone_angle # deg
# angle_to_z_axis = 0.2 * loss_cone_angle  # deg
# angle_to_z_axis = 0.5 * loss_cone_angle  # deg
# angle_to_z_axis = 0.7 * loss_cone_angle # deg
# angle_to_z_axis = 0.95 * loss_cone_angle # deg
# angle_to_z_axis = 0.99 * loss_cone_angle # deg
# angle_to_z_axis = loss_cone_angle # deg
# angle_to_z_axis = 1.01 * loss_cone_angle # deg
# angle_to_z_axis = 1.10 * loss_cone_angle # deg
# angle_to_z_axis = 1.5 * loss_cone_angle # deg
# angle_to_z_axis = 20
# angle_to_z_axis = 30
# angle_to_z_axis = 45
# angle_to_z_axis = 60
# angle_to_z_axis = 80
# angle_to_z_axis = 135
# angle_to_z_axis = 160
# angle_to_z_axis = 180
# angle_to_z_axis = 90
# angle_to_z_axis = 70
# angle_to_z_axis = 60
# angle_to_z_axis = 80
# angle_to_z_axis = 180 - angle_to_z_axis  # going in the negative direction

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

# l = 0.1  # m (interaction length)
# l = 0.5  # m (interaction length)
# l = 1.0  # m (interaction length)
# l = 2.0  # m (interaction length)
# l = 5.0  # m (interaction length)
l = 10.0  # m (interaction length)
# l = 100.0  # m (interaction length)


cyclotron_radius = np.linalg.norm(v_0) / omega_cyclotron
print('cyclotron_radius = ' + str(cyclotron_radius) + ' m')

r_0 = 0
# r_0 = cyclotron_radius
# r_0 = 0.1 * l
# r_0 = 0.3 * l
# r_0 = 0.4 * l
# r_0 = 0.5 * l
z_0 = 0.0 * l
# z_0 = 0.2 * l
# z_0 = 0.5 * l
x_0 = np.array([r_0, 0, z_0])

z_0 = x_0[2]
# z_0 = 0

v_z = v_0[2]

# t_max = l / v_z
# t_max = 3 * l / v_z
# t_max = 5 * l / v_z
# t_max = 10 * l / v_z
# t_max = 5 * l / v_th
t_max = 10 * l / v_th
# t_max = 20 * l / v_z
# t_max = 30 * l / v_z
# t_max = 50 * l / v_z
# t_max = abs(t_max)
# t_max = min(t_max, 100 * tau_cyclotron)
# t_max = 1000 * tau_cyclotron
# t_max = 20 * tau_cyclotron
# t_max = 100 * tau_cyclotron
# t_max = 1000 * tau_cyclotron

# dt = tau_cyclotron / 50 / Rm
# dt = tau_cyclotron / 300
# dt = tau_cyclotron / 200
# dt = tau_cyclotron / 150
# dt = tau_cyclotron / 100
# dt = tau_cyclotron / 50
dt = tau_cyclotron / 20
# dt = tau_cyclotron / 40
# dt = tau_cyclotron / 10
# num_steps = 1000
num_steps = int(t_max / dt)
# num_steps = min(num_steps, 10000)
# num_steps = min(num_steps, 20000)
# num_steps = 10000

print('num_steps = ', num_steps)
print('t_max = ', num_steps * dt, 's')

# RF definitions
# E_RF = 0
# E_RF = 1  # kV/m
E_RF = 2  # kV/m
# E_RF = 3  # kV/m
# E_RF = 5  # kV/m
# E_RF = 10  # kV/m
# E_RF = 15  # kV/m
# E_RF = 20  # kV/m
# E_RF = 50  # kV/m
# E_RF = 60  # kV/m
# E_RF = 70  # kV/m
# E_RF = 80  # kV/m
# E_RF = 100  # kV/m
E_RF *= 1e3  # the SI units is V/m

if B0 == 0:  # pick a default
    anticlockwise = 1
else:
    anticlockwise = np.sign(B0)

phase_RF = 0
# phase_RF = np.pi / 4
# phase_RF = np.pi / 2
# phase_RF = np.pi
# phase_RF = 1.5 * np.pi

# RF_type = 'uniform'
RF_type = 'traveling'

if RF_type == 'uniform':
    omega = omega_cyclotron  # resonance
    # omega = Rm * omega_cyclotron  # resonance
    # omega *= 1.01 # off resonance
    # omega *= 1.02 # off resonance
    # omega *= 1.05 # off resonance
    # omega *= 1.1 # off resonance
    # omega *= 0.9 # off resonance
    # omega *= 0.5 # off resonance
    # omega *= 2.0 # off resonance
    # omega *= 0.1 # off resonance

    k = omega / c
    # k = omega / v_th
    # k = - omega / v_th
    # k = omega / (2 * v_th)
    # k = omega / v_z
    # k = - omega / v_z
    # k = - omega / c

elif RF_type == 'traveling':

    # alpha_detune = 1.1
    # alpha_detune = 1.5
    # alpha_detune = 2.0
    alpha_detune = 2.718
    omega = alpha_detune * omega_cyclotron  # resonance

    # v_RF = alpha_detune / (alpha_detune - 1) * np.abs(v_z)
    # v_RF = alpha_detune / (alpha_detune - 1) * v_z
    v_RF = alpha_detune / (alpha_detune - 1) * v_th
    k = omega / v_RF


def E_function(x, t):
    z = x[2]
    return E_RF * np.array([anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
                            np.cos(k * (z - z_0) - omega * t + phase_RF),
                            0])


def B_function(x, t):
    use_transverse_fields = True
    # use_transverse_fields = False

    # B_mirror = magnetic_field_constant(B0)
    B_mirror = np.array([0, 0, B0])
    # B_mirror = magnetic_field_logan(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)
    # B_mirror = magnetic_field_jaeger(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)
    # B_mirror = magnetic_field_post(x, B0, Rm, l, use_transverse_fields=use_transverse_fields)

    # B_RF = 1/c * k_hat cross E_RF
    # https://en.wikipedia.org/wiki/Sinusoidal_plane-wave_solutions_of_the_electromagnetic_wave_equation
    # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
    z = x[2]
    B_RF = E_RF / c * np.array([-np.sign(k) * np.cos(k * (z - z_0) - omega * t + phase_RF),
                                np.sign(k) * anticlockwise * np.sin(k * (z - z_0) - omega * t + phase_RF),
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
Ex = hist['E'][:, 0] / 1e3
Ey = hist['E'][:, 1] / 1e3
Ez = hist['E'][:, 2] / 1e3

v_r = np.sqrt(vx ** 2 + vy ** 2)
v_perp_loss_cone = v_norm * Rm ** (-0.5)

print('mean(v_r) = ' + str(np.mean(v_r)))
print('mean(v_z) = ' + str(np.mean(vz)))

### Plots
linewidth = 2

# plt.figure(1)
# plt.figure(num=1, figsize=(15, 6))
# plt.figure(num=1, figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.figure(num=1, figsize=(16, 5))
plt.subplot(1, 3, 1)
# plt.plot(t, x, label='x', linewidth=linewidth, color='b')
# plt.plot(t, y, label='y', linewidth=linewidth, color='g')
plt.plot(t, R, label='$R/r_{cyc}$', linewidth=linewidth, color='b')
plt.plot(t, z, label='$z/l$', linewidth=linewidth, color='r')
plt.legend()
plt.xlabel('t / $\\tau_{cyc}$')
# plt.ylabel('coordinate / $r_{cyc}$')
plt.title('$E_{RF}$=' + str(E_RF / 1e3) + 'kV/m, $\\theta$=' + '{:.2f}'.format(angle_to_z_axis) + '$\degree$')
plt.ylabel('coordinate (normalized)')
plt.grid(True)
plt.tight_layout()

# plt.figure(4)
# plt.subplot(1,2,2)
plt.subplot(1, 3, 3)
# plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
# plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')

plt.plot(t, v_r, label='$v_r$', linewidth=linewidth, color='b')
# plt.plot(t, v_perp_loss_cone, label='$v_{r,LC}$', linewidth=linewidth, linestyle='--', color='c')
plt.plot(t, vz, label='$v_z$', linewidth=linewidth, color='r')
# plt.plot(t, v_norm, label='$v_{norm}$', linewidth=linewidth, color='k')
plt.legend()
plt.xlabel('t / $\\tau_{cyc}$')
plt.ylabel('velocity / $v_{th}$')
plt.grid(True)
plt.tight_layout()
#
# plt.figure(2)
# plt.plot(x, y, linewidth=linewidth)
# plt.xlabel('x / $r_{cyc}$')
# plt.ylabel('y / $r_{cyc}$')
# plt.grid(True)
# plt.tight_layout()
#
# plt.figure(3)
# plt.plot(R, z, linewidth=linewidth)
# # plt.xlabel('R / $r_{cyc}$')
# # plt.ylabel('z / $r_{cyc}$')
# plt.xlabel('R (normalized)')
# plt.ylabel('z (normalized)')
# plt.grid(True)
# # plt.legend()
# plt.tight_layout()
#
# plt.figure(5)
# E_kin = 0.5 * v_norm ** 2.0
# # plt.plot(t, E / E[0], linewidth=linewidth)
# plt.plot(t, (E_kin - E_kin[0]) / E_kin[0] * 100, linewidth=linewidth, color='k')
# plt.xlabel('t / $\\tau_{cyc}$')
# # plt.ylabel('$E / E_0$')
# plt.ylabel('kinetic energy % change')
# plt.grid(True)
# plt.tight_layout()
# print('energy change: ' + str((E_kin[-1] - E_kin[0]) / E_kin[0] * 100) + '%')
#
# plt.figure(6)
# plt.plot(t, B[:, 0], label='$B_x$', linewidth=linewidth, color='b')
# plt.plot(t, B[:, 1], label='$B_y$', linewidth=linewidth, color='g')
# plt.plot(t, B[:, 0], label='$B_r$', linewidth=linewidth, color='b')
# plt.plot(t, B[:, 2], label='$B_z$', linewidth=linewidth, color='r')
# plt.xlabel('t / $\\tau_{cyc}$')
# plt.ylabel('Tesla')
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
#
# if plot3d_exists is True:
#     plt.figure(7)
# else:
#     fig = plt.figure(7)
#     ax = Axes3D(fig)
# ax.plot(x, y, z, linewidth=linewidth)
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# ax.set_xlabel('x (normalized)')
# ax.set_ylabel('y (normalized)')
# ax.set_zlabel('z (normalized)')
# ax.set_title('particle 3d trajectory')
# # plt.legend()
#
# plt.figure(8)
# plt.plot(t, Ex, label='$E_x$', linewidth=linewidth, color='b')
# plt.plot(t, Ey, label='$E_y$', linewidth=linewidth, color='r')
# # Er = np.sign(Ex) * np.sqrt(Ex ** 2 + Ey ** 2)
# # r = np.sqrt(x ** 2 + y ** 2)
# # Er = r / x * Ex
# # plt.plot(t, Er, label='$E_r$', linewidth=linewidth, color='k')
# plt.xlabel('t / $\\tau_{cyc}$')
# plt.ylabel('$kV/m$')
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
#

# plt.figure(9)
# plt.subplot(1, 2, 2)
plt.subplot(1, 3, 2)
E_kin = 0.5 * v_norm ** 2.0
plt.plot(t, E_kin / E_kin[0], label='$E/E_0$', linewidth=linewidth, color='k')
plt.xlabel('t / $\\tau_{cyc}$')
plt.ylabel('$E/E_0$')
plt.grid(True)
plt.tight_layout()
# plt.legend()
