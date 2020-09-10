import matplotlib.pyplot as plt
import numpy as np

from loop_fields.aux_functions import fermi_dirac_periodic_square
from loop_fields.loop_functions import get_current_loop_magnetic_field_cylindrical
from loop_fields.plot_functions import plot_loop_symbols, plot_magnetic_field_lines

plt.close('all')

plot_B_map = False
# plot_B_map = True

# loop over different current ratios
# current_ratio_list = np.array([2, 3, 4, 5])
# current_ratio_list = np.array([2, 5])
current_ratio_list = np.array([5])
for ind_current_ratio, current_ratio in enumerate(current_ratio_list):
    print('current_ratio = ' + str(current_ratio))

    # loop over different lambda of MMM
    solenoid_r = 1.0
    # lambda_MMM_list = np.array([1, 2, 3, 4, 8, 12, 16]) * solenoid_r
    # lambda_MMM_list = np.array([1, 5, 10]) * solenoid_r
    # lambda_MMM_list = np.array([1, 5, 10, 20, 30]) * solenoid_r
    # lambda_MMM_list = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30]) * solenoid_r
    # lambda_MMM_list = np.array([10]) * solenoid_r
    # lambda_MMM_list = np.array([4]) * solenoid_r
    lambda_MMM_list = np.array([6]) * solenoid_r
    Rm_list = lambda_MMM_list * 0

    for ind_lambda, lambda_MMM in enumerate(lambda_MMM_list):
        # print('lambda_MMM / r = ' + str(lambda_MMM / solenoid_r))

        dr = 0.1
        r = np.arange(-1, 1, dr)
        # r = np.array([0])
        dl = 0.1
        z = np.arange(-10, 60, dl)

        rr, zz = np.meshgrid(r, z, indexing='ij')
        # print('rr shape = ' + str(rr.shape))
        # print('zz shape = ' + str(zz.shape))

        n_r = len(r)
        n_z = len(z)

        ### Main section + MM
        z1 = min(z)
        z2 = max(z)
        z_solenoid_end = z1 + 0.4 * (z2 - z1)
        # num_main_cell_coils = 10
        # z_MMM_start = z_solenoid_end + solenoid_r / 2
        # z_MMM_start = z_solenoid_end + solenoid_r / 4
        z_MMM_start = z_solenoid_end
        num_main_cell_coils = int(np.round((z_solenoid_end - z1) / (solenoid_r / 2)))
        loop_radius_list_main_cell = [1 for i in range(num_main_cell_coils)]

        minimal_current = 1.0
        # minimal_current = 0
        loop_current_list_main_cell = [minimal_current for i in range(num_main_cell_coils)]
        z0_list_main_cell = np.ndarray.tolist(np.linspace(z1, z_solenoid_end, num_main_cell_coils))
        color_list_main_cell = ['b' for i in range(num_main_cell_coils)]
        scale_list_main_cell = [0.5 for i in range(num_main_cell_coils)]

        # num_MM_coils = 20
        # num_MM_coils = int(np.round((z2 - z_MMM_start) / lambda_MMM) * 10)
        # coils_per_wavelength = 5
        coils_per_wavelength = 10
        # coils_per_wavelength = 20
        # coils_per_wavelength = 40
        # coils_per_wavelength = 100
        # coils_per_wavelength = 500
        # coils_per_wavelength = 1000
        # num_MM_coils = max(int(np.round((z2 - z_MMM_start) / lambda_MMM) * coils_per_wavelength),
        #                    int(np.round((z2 - z_MMM_start) / (solenoid_r / 2))))
        num_MM_coils = int(np.round((z2 - z_MMM_start) / lambda_MMM) * coils_per_wavelength)
        # num_MM_coils = int(np.round((z2 - z_MMM_start) / (solenoid_r / 2)))
        print('num_MM_coils = ' + str(num_MM_coils))

        loop_radius_list_MM_cell = [1 for i in range(num_MM_coils)]
        color_list_MM_cell = ['r' for i in range(num_MM_coils)]
        # l_cell = np.linspace(0, 2, num_MM_coils)
        z_MMM_array = np.linspace(z_MMM_start, z2, num_MM_coils)
        # phase = 0
        # phase = np.pi / 4
        phase = np.pi / 2

        # minimal_current_MM = 1.0
        # minimal_current_MM = 0.5
        minimal_current_MM = 0

        # pos_frac = 0.5
        # pos_frac = 0.4
        # pos_frac = 0.3
        # pos_frac = 0.2
        pos_frac = 0.1
        # pos_frac = 0.05
        # pos_frac = 0.02
        # smoothing_length = lambda_MMM * 0.005
        smoothing_length = lambda_MMM * 0.001
        loop_current_list_MM_cell = minimal_current_MM + (current_ratio - minimal_current_MM) / 2 * (
                1 + fermi_dirac_periodic_square(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase,
                                                L=lambda_MMM, dx=smoothing_length, pos_frac=pos_frac))
        # loop_current_list_MM_cell = minimal_current_MM + (current_ratio - minimal_current_MM) / 2 * ( 1 + np.sin(-(z_MMM_array - z_MMM_start) / lambda_MMM * 2.0 * np.pi - phase) )
        scale_list_MM_cell = np.ndarray.tolist(0.5 * loop_current_list_MM_cell / np.max(loop_current_list_MM_cell))
        loop_current_list_MM_cell = np.ndarray.tolist(loop_current_list_MM_cell)
        # z0_list_MM_cell = np.ndarray.tolist(np.linspace(z_solenoid_end, z2, num_MM_coils))
        z0_list_MM_cell = np.ndarray.tolist(z_MMM_array)

        # loop_radius_list = loop_radius_list_main_cell
        # loop_current_list = loop_current_list_main_cell
        # z0_list = z0_list_main_cell

        loop_radius_list = loop_radius_list_main_cell + loop_radius_list_MM_cell
        loop_current_list = loop_current_list_main_cell + loop_current_list_MM_cell
        z0_list = z0_list_main_cell + z0_list_MM_cell
        color_list = color_list_main_cell + color_list_MM_cell
        scale_list = scale_list_main_cell + scale_list_MM_cell

        # start_points = None
        n_lines = 20
        # start_points = (np.array([z_MMM_start + np.zeros(n_lines), np.linspace(-1, 1, n_lines)]).transpose())
        start_points = None

        ######################
        B_r_tot, B_z_tot = 0, 0

        for i in range(len(loop_radius_list)):
            loop_radius = loop_radius_list[i]
            loop_current = loop_current_list[i]
            z0 = z0_list[i]

            B_r, B_z = get_current_loop_magnetic_field_cylindrical(rr, zz, z0=z0, loop_radius=loop_radius,
                                                                   loop_current=loop_current)

            B_r_tot += B_r
            B_z_tot += B_z

            if plot_B_map:
                plt.figure(3 + len(current_ratio_list) + ind_lambda)
                # scale = 1.0
                # scale = 0.5 * loop_current
                scale = 0.25 * loop_current
                # scale = loop_current / np.mean(loop_current_list)
                plot_loop_symbols(z0=z0, loop_radius=loop_radius, scale=scale_list[i], color=color_list[i])

        if plot_B_map:
            plt.figure(3 + len(current_ratio_list) + ind_lambda)
            plot_magnetic_field_lines(z, r, B_z_tot, B_r_tot,
                                      # title='B field, MMM phase = ' + str(phase / np.pi) + '$\pi$',
                                      title='current ratio = ' + str(current_ratio) + ', $\\lambda_{MMM} / r $= ' + str(
                                          lambda_MMM / solenoid_r),
                                      start_points=start_points)

        # calculate current ratio
        # I_max = np.max(loop_current_list_MM_cell)
        # I_min = np.min(loop_current_list_MM_cell)
        # current_ratio = I_max / I_min
        # print('current_ratio = ' + str(current_ratio))

        # plot current as function of position on axis
        plt.figure(2 + ind_current_ratio)
        plt.plot(z0_list, loop_current_list,
                 label='lambda_MMM / r = ' + str(lambda_MMM / solenoid_r) + ', pos_frac = ' + str(pos_frac))
        # label = 'lambda_MMM / r = ' + str(lambda_MMM / solenoid_r))
        plt.xlabel('z [m]', size=15)
        plt.ylabel('Current', size=15)
        plt.title('Current in coils, current ratio = ' + str(current_ratio), size=15)
        plt.tight_layout()
        plt.legend()
        plt.grid(True)

        # calculate mirror ratio Rm
        B_z_on_axis = B_z_tot[int(B_z_tot.shape[0] / 2), :]
        # inds_MMM_section = np.where(z > z_MMM_start)[0]
        # inds_MMM_section = np.where(z > 5)[0]
        z_MMM_center = np.mean([z_MMM_start, z2])
        MMM_length = z2 - z_MMM_start
        inds_MMM_section = np.where(np.abs(z - z_MMM_center) < 0.7 * MMM_length / 2)[0]
        B_max = np.max(B_z_on_axis[inds_MMM_section])
        B_min = np.min(B_z_on_axis[inds_MMM_section])
        Rm = B_max / B_min
        print('Rm = ' + str(Rm))
        Rm_list[ind_lambda] = Rm

        # plot magnetic field profile on axis
        B = B_z_tot[int(B_z_tot.shape[0] / 2), :]
        # B = B_z_tot[int(B_z_tot.shape[0] / 2), :] / B_z_tot[int(B_z_tot.shape[0] / 2), int(B_z_tot.shape[1] / 8)]
        plt.figure(3 + len(current_ratio_list) + ind_current_ratio)
        plt.plot(z, B,
                 # label='lambda_MMM / r = ' + str(lambda_MMM / solenoid_r))
                 label='lambda_MMM / r = ' + str(lambda_MMM / solenoid_r) + ', pos_frac = ' + str(
                     pos_frac) + ', Rm =' + str(Rm))

        # label = 'MMM phase = ' + str(phase / np.pi) + '$\pi$')
        # plt.plot(z[inds_MMM_section[0]], 8, 'o')
        # plt.plot(z[inds_MMM_section[-1]], 8, 'x')
        plt.xlabel('z [m]', size=15)
        # plt.ylabel('$B_z$ [T]', size=15)
        plt.ylabel('$B_z/B_{z,0}$', size=15)
        plt.title('Axial magnetic field on axis (normalized), current ratio = ' + str(current_ratio), size=15)
        plt.tight_layout()
        plt.legend()
        plt.grid(True)

plt.figure(1)
plt.plot(lambda_MMM_list, Rm_list, '-o', label='current ratio = ' + str(current_ratio), linewidth=2)
plt.xlabel('$\\lambda_{MMM}/r$', size=15)
plt.ylabel('$R_m$', size=15)
plt.title('Mirror Ratio $B_{max}/B_{min}$', size=15)
plt.tight_layout()
plt.legend()
plt.grid(True)
