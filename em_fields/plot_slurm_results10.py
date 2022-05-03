import pickle

import warnings

warnings.filterwarnings("error")

import numpy as np

from em_fields.slurm_functions import get_script_evolution_slave_fenchel

evolution_slave_fenchel_script = get_script_evolution_slave_fenchel()

import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 10})
# plt.rcParams["figure.facecolor"] = 'white'
# plt.rcParams["axes.facecolor"] = 'green'

plt.close('all')

save_dir = '/Users/talmiller/Downloads/single_particle/'
# save_dir += '/set26_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'
save_dir += '/set27_B0_1T_l_3m_Post_Rm_3_first_cell_center_crossing/'

RF_type = 'electric_transverse'
# E_RF_kVm = 1 # kV/m
# E_RF_kVm = 10  # kV/m
# E_RF_kVm = 30  # kV/m
E_RF_kVm = 50  # kV/m
# E_RF_kVm = 100  # kV/m

# RF_type = 'magnetic_transverse'
B_RF = 0.05  # T
# B_RF = 0.1  # T

use_RF = True
# use_RF = False

absolute_velocity_sampling_type = 'maxwell'
# absolute_velocity_sampling_type = 'const_vth'
r_0 = 0
# r_0 = 1.0
# r_0 = 3.0

# alpha_loop_list = np.round(np.linspace(0.9, 1.1, 11), 2)  # set26
# beta_loop_list = np.round(np.linspace(0, 1, 11), 11)

alpha_loop_list = np.round(np.linspace(0.9, 1.1, 21), 2)  # set27
beta_loop_list = np.round(np.linspace(-1, 1, 21), 2)

# for ind_beta, beta_RF in enumerate(beta_loop_list):
#     for ind_alpha, alpha_RF in enumerate(alpha_loop_list):
# ind_alpha = 0
# ind_alpha = 4
# ind_alpha = 5
# ind_alpha = 7
# ind_beta = 5
# ind_beta = 4
# ind_beta = 3
# alpha = alpha_loop_list[ind_alpha]
# beta = beta_loop_list[ind_beta]
# alpha = 1.05
# alpha = 0.9
alpha = 0.95
# alpha = 0.97
# alpha = 0.98
# alpha = 0.99
# alpha = 1.0
# alpha = 1.01
# alpha = 1.02
# alpha = 1.04
# alpha = 1.05
# alpha = 1.1
# beta = 0.0
# beta = -0.1
# beta = -0.2
# beta = -0.3
# beta = -0.4
# beta = -0.5
# beta = -0.7
beta = -1.0
# beta = 0.5
# beta = 1.0

if use_RF is False:
    title = 'without RF'
elif RF_type == 'electric_transverse':
    title = '$E_{RF}$=' + str(E_RF_kVm) + 'kV/m, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
elif RF_type == 'magnetic_transverse':
    title = '$B_{RF}$=' + str(B_RF) + 'T, $\\alpha$=' + str(alpha) + ', $\\beta$=' + str(beta)
print(title)

set_name = ''
if use_RF is False:
    set_name += 'without_RF'
else:
    if RF_type == 'electric_transverse':
        set_name += 'ERF_' + str(E_RF_kVm)
    elif RF_type == 'magnetic_transverse':
        set_name += 'BRF_' + str(B_RF)
    set_name += '_alpha_' + str(alpha)
    set_name += '_beta_' + str(beta)
if absolute_velocity_sampling_type == 'const_vth':
    set_name = 'const_vth_' + set_name
if r_0 > 0:
    set_name = 'r0_' + str(r_0) + '_' + set_name

save_dir_curr = save_dir + set_name

# load runs data
data_dict_file = save_dir_curr + '.pickle'
with open(data_dict_file, 'rb') as fid:
    data_dict = pickle.load(fid)
settings_file = save_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
field_dict_file = save_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)

for key in data_dict.keys():
    data_dict[key] = np.array(data_dict[key])

dt = data_dict['t'][:, 1] - data_dict['t'][:, 0]
z = data_dict['z'][:, 1]
v = data_dict['v'][:, 1]
v0 = data_dict['v'][:, 0]
vt = data_dict['v_transverse'][:, 1]
vt0 = data_dict['v_transverse'][:, 0]
vz = data_dict['v_axial'][:, 1]
vz0 = data_dict['v_axial'][:, 0]
# theta0 = 360 / (2 * np.pi) * np.arcsin(vt0 / v0) * np.sign(vz0)
theta0 = np.mod(360 / (2 * np.pi) * np.arctan(vt0 / vz0), 180)
theta_LC = 360 / (2 * np.pi) * np.arcsin(1 / np.sqrt(field_dict['Rm']))
# theta_bins = np.linspace(-90, 90, 31)
# theta_bins = np.linspace(-90, 90, 31) + 180 / 30.0
# theta_bin_width = theta_bins[1] - theta_bins[0]
# theta_bin_width = 3
theta_bin_width = 5
# theta_bins = np.arange(-90, 90, theta_bin_width)
# theta_bins = np.delete(theta_bins, np.where(theta_bins == 0))
# theta_bins = np.arange(theta_bin_width / 2.0, 90, theta_bin_width)
# theta_bins = np.append(-np.flip(theta_bins), theta_bins)
theta_bins = np.arange(theta_bin_width / 2.0, 180, theta_bin_width)

# normalize_by_tfin = True
normalize_by_tfin = False

# fig, (axs) = plt.subplots(2, 2, figsize=(10,6))
fig, (axs) = plt.subplots(3, 2, figsize=(10, 9))
ax = axs[0, 0]
# plt.figure(1)
ax.scatter(vz0 / settings['v_th'], vt0 / settings['v_th'], label='ini', alpha=0.5, color='b')
ax.scatter(vz / settings['v_th'], vt / settings['v_th'], label='fin', alpha=0.5, color='r')
ax.set_xlabel('$v_z / v_{th}$')
ax.set_ylabel('$v_{\\perp} / v_{th}$')
ax.set_title(title)
ax.legend()
ax.grid(True)


def plot_dist(y, ax, ylabel):
    # plt.figure(fig_num)
    # ax = plt.gca()
    # df = pd.DataFrame({
    #     'X': theta0,
    #     'Y': y,
    # })
    # df['Xbins'] = np.digitize(df.X, theta_bins)
    # df['Ymean'] = df.groupby('Xbins').Y.transform('mean')
    # # df['Ystd'] = df.groupby('Xbins').Y.transform('std')
    # df.plot(kind='scatter', x='X', y='Ymean', color='k', label='pandas grouping', ax=ax)

    y_binned = 0 * theta_bins
    y_binned_std = 0 * theta_bins
    y_binned_upper = 0 * theta_bins
    y_binned_lower = 0 * theta_bins
    for i, theta_bin in enumerate(theta_bins):
        # print(i, theta_bin)
        inds = np.where(np.abs(theta0 - theta_bin) < theta_bin_width / 2.0)
        y_curr_bin_array = y[inds]
        # y_binned[i] = np.mean(y_curr_bin_array)
        y_binned[i] = np.percentile(y_curr_bin_array, 50)
        y_binned_std[i] = np.std(y_curr_bin_array)
        # prctile = 5
        prctile = 10
        # prctile = 33
        y_binned_upper[i] = np.percentile(y_curr_bin_array, 100 - prctile)
        y_binned_lower[i] = np.percentile(y_curr_bin_array, prctile)
    # plt.errorbar(theta_bins, y_binned, yerr=y_binned_std, label='manual binning', color='g')
    # plt.fill_between(theta_bins, y_binned + y_binned_std, y_binned - y_binned_std, color='b', alpha=0.5)
    # ax.fill_between(theta_bins, y_binned_upper, y_binned_lower, color='b', alpha=0.5)
    # ax.plot(theta_bins, y_binned, label='manual binning', color='b')

    ax.scatter(theta0, y, color='b', alpha=0.5)
    #
    # # ymin = np.min(y_binned - y_binned_std)
    # # ymax = np.max(y_binned + y_binned_std)
    ymin = np.min(y_binned_lower)
    ymax = np.max(y_binned_upper)
    ax.vlines(theta_LC, ymin, ymax, color='k', linestyle='--')
    ax.vlines(180 - theta_LC, ymin, ymax, color='k', linestyle='--')
    # ax.vlines(-theta_LC, ymin, ymax, color='k', linestyle='--')
    ax.set_xlabel('$\\theta$ [deg]')
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    # ax.legend()
    ax.grid(True)

    # y_binned_stats = scipy.stats.binned_statistic(theta0, y, statistic='mean', bins=theta_bins)
    # y_std_binned_stats = scipy.stats.binned_statistic(theta0, y, statistic='std', bins=theta_bins)
    # plt.errorbar(theta_bins, y_binned_stats, yerr=y_std_binned_stats, label='stats', color='r')

    return


y = (v - v0) / settings['v_th']
ylabel = '$\\Delta v / v_{th} $'
if normalize_by_tfin is True:
    y /= dt / (settings['l'] / settings['v_th'])
    ylabel = '$\\Delta v / v_{th}  / (t_{fin} v_{th} / l)$'
ax = axs[0, 1]
plot_dist(y, ax, ylabel=ylabel)

y = (vt - vt0) / settings['v_th']
ylabel = '$\\Delta v_{\\perp} / v_{th}$'
if normalize_by_tfin is True:
    y /= dt / (settings['l'] / settings['v_th'])
    ylabel = '$\\Delta v_{\\perp} / v_{th} / (t_{fin} v_{th} / l)$'
ax = axs[1, 0]
plot_dist(y, ax, ylabel=ylabel)

y = (vt / v - vt0 / v0)
ylabel = '$\\Delta ( v_{\\perp} / v )$'
if normalize_by_tfin is True:
    y /= dt / (settings['l'] / settings['v_th'])
    ylabel = '$\\Delta ( v_{\\perp} / v ) / (t_{fin} v_{th} / l)$'
ax = axs[1, 1]
plot_dist(y, ax, ylabel=ylabel)

ax = axs[2, 0]
plot_dist(z / settings['l'], ax, ylabel='$z_{fin}/l$')

ax = axs[2, 1]
plot_dist(dt / (settings['l'] / settings['v_th']), ax, ylabel='$t_{fin} v_{th} / l$')

plt.tight_layout()
