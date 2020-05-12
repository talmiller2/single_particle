import numpy as np


def fermi_dirac_square(x, x0=0, L=1, dx=0.02):
    fd = 1 / ((1 + np.exp(- (x - x0) / dx)) * (1 + np.exp((x - x0 - L) / dx)))
    return 2 * (fd - 1.0 / 2)


# def fermi_dirac_periodic_square(x, L=1, dx=0.02):
#     x_mod = np.mod(x, L)
#     x_mod_half = np.mod(x, L/2)
#     sign_half = - np.sign(x_mod - L/2) # positive in the first half of wavelength
#     return sign_half * fermi_dirac_square(x_mod_half, x0=0, L=L / 2, dx=dx)

def fermi_dirac_periodic_square(x, L=1, dx=0.02, pos_frac=0.5):
    x_mod = np.mod(x, L)
    L1 = pos_frac * L
    L2 = L - L1
    sign_pos_frac = - np.sign(x_mod - L1)  # positive in first part of wavelength L
    # define x relative to the first or second part of the wavelength L
    x_rel = (1 + sign_pos_frac) / 2 * x_mod + (1 - sign_pos_frac) / 2 * (x_mod - L1)
    L_rel = (1 + sign_pos_frac) / 2 * L1 + (1 - sign_pos_frac) / 2 * L2
    return sign_pos_frac * fermi_dirac_square(x_rel, x0=0, L=L_rel, dx=dx)
