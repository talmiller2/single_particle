import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.close('all')

# z = np.linspace(-10, 10, 1000)
z = np.linspace(-2, 2, 1000)

B0 = 1
Rm = 3
Bmax = B0 * Rm

z_main = 0
dL = 0.1

# Bz_main = B0 + (Bmax - B0) / (1 + np.exp(-(z - z_main)/ dL))
Bz_main = B0 + 0 * z

l = 1
z_MMM_list = [z_main + i * l for i in range(10)]


# z_MMM_list = [1, 5]
# z_MMM_list = [1]

def get_Bz_MMM_cell(z, z_list, frac):
    Bz_MMM = 0
    main_cell_wall = 1 / (1 + np.exp(- (z - z_main) / dL))
    for zi in z_list:
        # Bz_MMM += (Bmax - B0) / (1 + np.exp(-(z - zi)/ dL)) / (1 + np.exp((z - zi - dL)/ dL))
        Bz_MMM += (Bmax - B0) * np.exp(-((z - zi + frac * l) / dL) ** 2) * main_cell_wall
    return Bz_MMM


# frac_list = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
# frac_list = [0, 0.05, 0.1, 0.15, 0.2]
frac_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colors = cm.rainbow(np.linspace(0, 1, len(frac_list)))

plt.figure(figsize=(7, 5))
plt.vlines(z_main, 0, Bmax, label='main cell wall', colors='grey', linestyles='--')

for frac, color in zip(frac_list, colors):
    # Bz_MMM = B0 + (Bmax - B0) / (1 + np.exp(-(z - z_main)/ dL))
    Bz_MMM = get_Bz_MMM_cell(z, z_MMM_list, frac)
    # Bz_tot = Bz_main + Bz_MMM
    # plt.plot(z, Bz_MMM, color=color, label='t=' + str(frac))
    plt.plot(z, Bz_main + Bz_MMM, color=color, label='$t/\\tau$=' + str(frac))

# plt.plot(z, Bz_main)
# plt.plot(z, Bz_MMM)
# plt.plot(z, Bz_MMM_2)
# plt.plot(z, Bz_tot)
plt.xlabel('z [m]')
plt.ylabel('$B_z$ [T]')
plt.title('fields of main cell + MMM section')
plt.grid(True)
plt.legend()
plt.tight_layout()
