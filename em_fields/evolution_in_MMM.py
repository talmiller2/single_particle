import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from em_fields.default_settings import define_default_settings
from em_fields.em_functions import evolve_particle_in_em_fields, get_thermal_velocity, get_cyclotron_angular_frequency
from em_fields.magnetic_forms import *

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelpad': 15})
plt.rcParams.update({'lines.linestyle': '-'})
# plt.rcParams.update({'lines.linestyle': '--'})

plt.close('all')


settings = define_default_settings()
m = settings['mi']
q = settings['Z_ion'] * settings['e']  # Coulomb
T_eV = 1e3
kB_eV = settings['kB_eV']
v_th = get_thermal_velocity(T_eV, m, kB_eV)

Rm = 3.0
print('Rm = ' + str(Rm))

loss_cone_angle = np.arcsin(Rm ** (-0.5)) * 360 / (2 * np.pi)
print('loss_cone_angle = ' + str(loss_cone_angle))

# B0 = 0 # Tesla
# B0 = 0.1  # Tesla
# B0 = 0.2  # Tesla
B0 = 1.0  # Tesla
# B0 = -1.0 # Tesla

omega_cyclotron = get_cyclotron_angular_frequency(q, B0, m)
tau_cyclotron = 2 * np.pi / omega_cyclotron

# angle_to_z_axis = 0  # deg
# angle_to_z_axis = 0.5 * loss_cone_angle # deg
# angle_to_z_axis = 0.95 * loss_cone_angle # deg
# angle_to_z_axis = 0.99 * loss_cone_angle # deg
# angle_to_z_axis = loss_cone_angle # deg
angle_to_z_axis = 1.01 * loss_cone_angle  # deg
# angle_to_z_axis = 1.5 * loss_cone_angle # deg
# angle_to_z_axis = 1.10 * loss_cone_angle # deg
# angle_to_z_axis = 1.2 * loss_cone_angle # deg
# # angle_to_z_axis = 2 * loss_cone_angle # deg
# angle_to_z_axis = 90 # deg
print('angle_to_z_axis = ' + str(angle_to_z_axis) + ' degrees')

angle_to_z_axis_rad = angle_to_z_axis / 360 * 2 * np.pi
v_0 = np.array([0, np.sin(angle_to_z_axis_rad), np.cos(angle_to_z_axis_rad)])
v_0 *= v_th

# l = 0.002  # m (interaction length)
# l = 0.01  # m (interaction length)
# l = 0.05  # m (interaction length)
l = 0.1  # m (interaction length)
# l = 0.5  # m (interaction length)
# l = 0.8  # m (interaction length)
# l = 1.0  # m (interaction length)
# l = 2.0  # m (interaction length)
# l = 5.0  # m (interaction length)
# l = 100.0  # m (interaction length)
v_z = np.abs(v_0[2])
# t_max = l / v_z
# t_max = 2 * l / v_z
t_max = 5 * l / v_z
# t_max = 7 * l / v_z
# t_max = 10 * l / v_z
# t_max = min(t_max, 100 * tau_cyclotron)

# cyclotron_radius = np.linalg.norm(v_0) / omega_cyclotron
v0_r = np.sqrt(v_0[0] ** 2 + v_0[1] ** 2)
cyclotron_radius = v0_r / omega_cyclotron
# x_0 = np.array([0, 0, 0])
# x_0 = cyclotron_radius * np.array([1, 0, 0])
# x_0 = cyclotron_radius * np.array([0, 0, 0])
# x_0 = np.array([0, 0, 0])
# x_0 = np.array([0, 0, 0.5 * l])
x_0 = np.array([cyclotron_radius, 0, 0.5 * l])
# x_0 = np.array([0, 0, 0])


# dt = tau_cyclotron / 50 / Rm
# dt = tau_cyclotron / 200
# dt = tau_cyclotron / 20
# dt = tau_cyclotron / 30
dt = tau_cyclotron / 50
# dt = tau_cyclotron / 100
# num_steps = 1000
num_steps = int(t_max / dt)
# num_steps = min(num_steps, 3000)

print('num_steps = ', num_steps)
print('t_max = ', num_steps * dt, 's')

# MMM velocity
U = 0 * v_th
# U = 0.1 * v_th
# U = 0.15 * v_th
# U = 0.2 * v_th
# U = 0.25 * v_th
# U = 0.3 * v_th
# U = 0.35 * v_th
# U = 0.38 * v_th
# U = 0.4 * v_th
# U = 0.5 * v_th
print('U / vth in lab frame = ' + str(U / v_th))
v_ref = U

# mode = 'lab_frame'
mode = 'lab_frame_shift_solution_to_MMM'
# mode = 'MMM_frame'
print('mode: ' + mode)

# test MMM reference frame invariance
if mode == 'MMM_frame':
    v_ref = U
    v_0 += np.array([0, 0, v_ref])
    U = 0


def B_function(x, t):
    use_transverse_fields = True
    # use_transverse_fields = False

    # B_mirror = magnetic_field_constant(B0)
    B_mirror = magnetic_field_logan(x, B0, Rm, l, z0=-U * t, use_transverse_fields=use_transverse_fields)
    # B_mirror = magnetic_field_jaeger(x, B0, Rm, l, z0=-U*t, use_transverse_fields=use_transverse_fields)
    # B_mirror = magnetic_field_post(x, B0, Rm, l, z0=-U*t, use_transverse_fields=use_transverse_fields)
    # B_mirror = magnetic_field_linear(x, B0, l, z0=-U*t, use_transverse_fields=use_transverse_fields)
    return B_mirror


# def E_function(x, t):
#     return np.array([0, 0, 0])

def E_function(x, t):
    B = B_function(x, t)
    r = get_radius(x)
    B_r = r / x[0] * B[0]
    E_theta = U * B_r
    theta = np.arctan2(x[1], x[0])
    return E_theta * np.array([np.sin(theta), -np.cos(theta), 0])


hist = evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function,
                                    num_steps=num_steps, q=q, m=m, return_fields=True)
t = hist['t']
x = hist['x'][:, 0]
y = hist['x'][:, 1]
z = hist['x'][:, 2]

vx = hist['v'][:, 0]
vy = hist['v'][:, 1]
vz = hist['v'][:, 2]

Bx = hist['B'][:, 0]
By = hist['B'][:, 1]
Bz = hist['B'][:, 2]

Ex = hist['E'][:, 0] / 1e3
Ey = hist['E'][:, 1] / 1e3
Ez = hist['E'][:, 2] / 1e3

# When using MMM, change reference frame to the MMM rest frame
if mode == 'lab_frame_shift_solution_to_MMM':
    vz = vz + v_ref
    z = z + v_ref * t

R = np.sqrt(x ** 2 + y ** 2)
v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
vr = np.sqrt(vx ** 2 + vy ** 2)
v_perp_loss_cone_static = v_norm * Rm ** (-0.5)
v_perp_loss_cone_dynamic = np.sqrt(vr ** 2 + (vz + U) ** 2) * Rm ** (-0.5)
E_kin = 0.5 * v_norm ** 2.0

# scale all observables
# t /= tau_cyclotron
t /= (l / v_th)
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
vr /= v_th
U /= v_th
v_perp_loss_cone_static /= v_th
v_perp_loss_cone_dynamic /= v_th

### Plots
linewidth = 2

plt.figure(1)
# plt.subplot(1,2,1)
# plt.plot(t, x, label='$x/r_{cyc}$', linewidth=linewidth, color='b')
plt.plot(t, y, label='$y/r_{cyc}$', linewidth=linewidth, color='g')
plt.plot(t, R, label='$R/r_{cyc}$', linewidth=linewidth, color='b')
plt.plot(t, z, label='$z/l$', linewidth=linewidth, color='r')
plt.legend()
# plt.xlabel('t / $\\tau_{cyc}$')
plt.xlabel('t / $(l/v_{th})$')
# plt.ylabel('coordinate / $r_{cyc}$')
plt.ylabel('coordinate (normalized)')
plt.grid(True)
plt.tight_layout()

plt.figure(4)
# plt.subplot(1,2,2)
# plt.plot(t, vx, label='$v_x$', linewidth=linewidth, color='b')
# plt.plot(t, vy, label='$v_y$', linewidth=linewidth, color='g')
plt.plot(t, vr, label='$v_r$', linewidth=linewidth, color='b')
plt.plot(t, v_perp_loss_cone_static, label='$v_{r,LC,static}$', linewidth=linewidth, color='c')
plt.plot(t, v_perp_loss_cone_dynamic, label='$v_{r,LC,dynamic}$', linewidth=linewidth, color='m')
plt.plot(t, vz, label='$v_z$', linewidth=linewidth, color='r')
plt.plot(t, v_norm, label='$v_{norm}$', linewidth=linewidth, color='k')
plt.legend()
# plt.xlabel('t / $\\tau_{cyc}$')
plt.xlabel('t / $(l/v_{th})$')
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
# plt.plot(t, E_kin, linewidth=linewidth)
# plt.plot(t, E_kin / E_kin[0], linewidth=linewidth)
plt.plot(t, (E_kin - E_kin[0]) / E_kin[0] * 100, linewidth=linewidth)
# plt.xlabel('t / $\\tau_{cyc}$')
plt.xlabel('t / $(l/v_{th})$')
# plt.ylabel('$E$')
# plt.ylabel('$E / E_0$')
plt.ylabel('kinetic energy % increase')
plt.grid(True)
plt.tight_layout()
print('energy change: ' + str((E_kin[-1] - E_kin[0]) / E_kin[0] * 100) + '%')

plt.figure(6)
plt.plot(t, Bx, label='$B_r$', linewidth=linewidth, color='b')
plt.plot(t, Bz, label='$B_z$', linewidth=linewidth, color='r')
plt.xlabel('t / $\\tau_{cyc}$')
plt.ylabel('Tesla')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.figure(7)
plt.plot(t, Ex, label='$E_x$', linewidth=linewidth, color='b')
plt.plot(t, Ey, label='$E_y$', linewidth=linewidth, color='r')
# Er = np.sign(Ex) * np.sqrt(Ex ** 2 + Ey ** 2)
# r = np.sqrt(x ** 2 + y ** 2)
# Er = r / x * Ex
# plt.plot(t, Er, label='$E_r$', linewidth=linewidth, color='k')
plt.xlabel('t / $\\tau_{cyc}$')
plt.ylabel('$kV/m$')
plt.grid(True)
plt.tight_layout()
plt.legend()

fig = plt.figure(8)
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
