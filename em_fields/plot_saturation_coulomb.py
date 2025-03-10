import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})
plt.close('all')


def sol_rate_eqs(t, alpha, N_r0, N_l0, N_c0):
    """
    general analytic solution of the ODE system:
    d/dt n_c = (n_r + n_l) * (1 - 2 * alpha) - 2 * alpha * n_c
    d/dt n_r = (n_c + n_l) * alpha - (1 - alpha) * n_r
    d/dt n_l = (n_c + n_r) * alpha - (1 - alpha) * n_l
    """
    N_c = (N_c0 + N_r0 + N_l0) * (1 - 2 * alpha) + (2 * alpha * N_c0 + (2 * alpha - 1) * (N_r0 + N_l0)) * np.exp(-t)
    N_r = (N_c0 + N_r0 + N_l0) * alpha + (- alpha * N_c0 + (1 - alpha) * N_r0 - alpha * N_l0) * np.exp(-t)
    N_l = (N_c0 + N_r0 + N_l0) * alpha + (- alpha * N_c0 + (1 - alpha) * N_l0 - alpha * N_r0) * np.exp(-t)
    return N_r, N_l, N_c


# plot theoretic saturation curves for isotropic Coulomb scattering
Rm = 3
loss_cone_angle_rad = np.arcsin(1 / np.sqrt(Rm))
alpha = omega_loss_cone_fraction = np.sin(loss_cone_angle_rad / 2) ** 2
alpha_c = 1 - 2 * alpha

t = np.linspace(0, 3, 100)
tau = 1

N_rc = N_lc = alpha_c * (1 - np.exp(-t / tau))
N_cr = N_cl = N_rl = N_lr = alpha * (1 - np.exp(-t / tau))

# plt.figure()

N_r, N_l, N_c = sol_rate_eqs(t, alpha, N_r0=1, N_l0=0, N_c0=0)
N_rc_full = N_c / 1
# plt.plot(t, N_r, 'b', label='N_r for N_r0=1')
# plt.plot(t, N_l, 'g', label='N_r for N_r0=1')
# plt.plot(t, N_c, 'r', label='N_r for N_r0=1')

N_r, N_l, N_c = sol_rate_eqs(t, alpha, N_r0=0, N_l0=0, N_c0=1)
N_cr_full = N_r / 1
# plt.plot(t, N_r, '--b', label='N_r for N_c0=1')
# plt.plot(t, N_l, '--g', label='N_r for N_c0=1')
# plt.plot(t, N_c, '--r', label='N_r for N_c0=1')

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
plt.plot(t / tau, N_rc, color='b', label='$N_{rc}=N_{lc}$')
plt.plot(t / tau, N_cr, color='g', label='$N_{cr}=N_{cl}=N_{rl}=N_{lr}$')
# plt.plot(t / tau, N_rc_full, linestyle='--', color='cyan', label='$N_{rc}=N_{lc}$')
# plt.plot(t / tau, N_cr_full, linestyle='--', color='orange', label='$N_{cr}=N_{cl}=N_{rl}=N_{lr}$')
# plt.plot(t / tau, N_rc, color='b', label='$\\bar{N}_{rc}$')
# plt.plot(t / tau, N_cr, color='g', label='$\\bar{N}_{cr}$')
# plt.plot(t / tau, N_lc, color='r', label='$\\bar{N}_{lc}$')
# plt.plot(t / tau, N_cl, color='orange', label='$\\bar{N}_{cl}$')
# plt.plot(t / tau, N_rl, color='k', label='$\\bar{N}_{rl}$')
# plt.plot(t / tau, N_lr, color='brown', label='$\\bar{N}_{lr}$')
# ax.set_xlabel('t/($l/v_{th}$)', fontsize=12)
ax.set_xlabel('$t/\\tau_s$', fontsize=15)
ax.set_ylabel('$\\frac{ \\Delta N_{i \\rightarrow j} }{ N_{i,0} }$', fontsize=25, rotation=0, labelpad=30)
# ax.legend(loc='lower right', fontsize=15)
ax.legend(fontsize=15)
ax.grid(True)
ax.set_title('Populations evolution in isotropic Coulomb scattering ($R_m=' + str(Rm) + '$)')
fig.set_layout_engine(layout='tight')
