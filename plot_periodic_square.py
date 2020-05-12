import matplotlib.pyplot as plt
import numpy as np

from aux_functions import fermi_dirac_periodic_square

dx = 0.01

x = np.linspace(-2, 2, 1000)
plt.figure()
plt.plot(x, np.sin(x * 2 * np.pi), label='sine')
# plt.plot(x, fermi_dirac_square(x, dx=dx), label='Fermi-Dirac square')
plt.plot(x, fermi_dirac_periodic_square(x, dx=dx, pos_frac=0.5), label='Fermi-Dirac periodic square (frac=0.5)')
plt.plot(x, fermi_dirac_periodic_square(x, dx=dx, pos_frac=0.2), label='Fermi-Dirac periodic square (frac=0.2)')
plt.legend(prop={'size': 15})
plt.grid()
