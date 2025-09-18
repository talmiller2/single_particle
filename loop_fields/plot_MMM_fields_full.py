import matplotlib.pyplot as plt
import numpy as np

from em_fields.MMM_field_forms import get_MMM_magnetic_field, get_MMM_electric_field
from em_fields.default_settings import define_default_settings, define_default_field

axes_label_size = 12
# axes_label_size = 18
title_fontsize = 12

plt.close('all')

# z = np.linspace(-10, 10, 1000)
z = np.linspace(-4, 4, 1000)
# z = np.linspace(-5, 5, 1000)

## definitions
settings = define_default_settings()
field_dict = {}
field_dict['use_static_main_cell'] = False
# field_dict['use_static_main_cell'] = True
field_dict['Rm'] = 6
# field_dict['Rm'] = 5
field_dict['Rm_main'] = 3
field_dict = define_default_field(settings, field_dict)
# field_dict['U_MMM'] = 1e-4
field_dict['U_MMM'] = 0.1 * settings['v_th']
# field_dict['U_MMM'] = 1.0 * settings['v_th']
# tau = settings['l'] / settings['v_th']
tau = settings['l'] / field_dict['U_MMM']

# frac_list = [0]
# frac_list = [0, 0.5, 1.0]
frac_list = [0, 0.25, 0.5, 0.75]
# frac_list = [0, 0.2, 0.4, 0.6, 0.8]
# frac_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# colors = cm.rainbow(np.linspace(0, 1, len(frac_list)))
colors = ['b', 'g', 'orange', 'r']


def plot_wall_lines():
    plt.axvline(field_dict['MMM_z_wall'], label='main cell wall', color='grey', linestyle='--')
    plt.axvline(- field_dict['MMM_z_wall'], color='grey', linestyle='--')
    return


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plot_wall_lines()
for frac, color in zip(frac_list, colors):
    Bz_MMM = []
    for z_curr in z:
        x = [0, 0, z_curr]
        t = frac * tau
        Bz_MMM += [get_MMM_magnetic_field(x, t, **field_dict)[2]]
    plt.plot(z, Bz_MMM, color=color, label='$t/\\tau$=' + str(frac))
plt.xlabel('z [m]', fontsize=axes_label_size)
plt.ylabel('$B_z$ [T]', fontsize=axes_label_size)
plt.title('Axial magnetic field of MMM', fontsize=title_fontsize)
plt.grid(True)
plt.legend(fontsize=axes_label_size, loc='upper left')
plt.tight_layout()

fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
plot_wall_lines()
r = 0.1  # [m]
# r = 0.2 # [m]
for frac, color in zip(frac_list, colors):
    Br_MMM = []
    for z_curr in z:
        x = [r, 0, z_curr]
        t = frac * tau
        Bz_MMM = get_MMM_magnetic_field(x, t, **field_dict)
        # Br_MMM += [np.sqrt(Bz_MMM[0] ** 2 + Bz_MMM[1] ** 2)]
        Br_MMM += [Bz_MMM[0]]
    plt.plot(z, Br_MMM, color=color, label='$t/\\tau$=' + str(frac))
plt.xlabel('z [m]', fontsize=axes_label_size)
# plt.ylabel('$B_r$ [T]')
# plt.title('Radial magnetic field of MMM at r=' + str(r) + '[m]')
plt.ylabel('$B_x$ [T]', fontsize=axes_label_size)
plt.title('Transverse magnetic field of MMM at r=' + str(r) + '[m]', fontsize=title_fontsize)
plt.grid(True)
plt.legend(fontsize=axes_label_size)
plt.tight_layout()

fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
plot_wall_lines()
for frac, color in zip(frac_list, colors):
    Etheta_MMM = []
    for z_curr in z:
        x = [r, 0, z_curr]
        t = frac * tau
        Etheta_MMM += [get_MMM_electric_field(x, t, **field_dict)[1] / 1e3]
    plt.plot(z, Etheta_MMM, color=color, label='$t/\\tau$=' + str(frac))
plt.xlabel('z [m]', fontsize=axes_label_size)
# plt.ylabel('$E_\\theta$ [kV/m]')
# plt.title('Tangential electric field of MMM at r=' + str(r) + '[m]')
plt.ylabel('$E_y$ [kV/m]', fontsize=axes_label_size)
plt.title('Transverse electric field of MMM at r=' + str(r) + '[m]', fontsize=title_fontsize)
plt.grid(True)
plt.legend(fontsize=axes_label_size)
plt.tight_layout()

# ### saving figures
# fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2026/pics/'
# file_name = 'full_MMM_axial_B_field'
# if field_dict['use_static_main_cell']:
#     file_name += '_with_static_maincell'
# fig.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)
