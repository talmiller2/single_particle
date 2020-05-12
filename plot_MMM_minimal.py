import matplotlib.pyplot as plt
import numpy as np

from magnetic_field_functions import get_current_loop_magnetic_field_cylindrical

plt.rcParams.update({'font.size': 15})

I0 = 1.0
solenoid_r = 1.0


def calc_mirror_ratio(lambda_MMM, DC_current, AC_current, negative_AC_current=0,
                      plot_B_map=False, plot_curves=False, fig_num=3):
    # define a grid
    dr = 0.1
    # r = np.arange(-1, 1, dr)
    # r = np.arange(-0.1, 0.1, dr)
    r = np.array([0])
    # dl = 0.1
    # z = np.arange(-10, 60, dl)
    dl = lambda_MMM / 100
    z = np.arange(-5 * lambda_MMM, 10 * lambda_MMM, dl)

    rr, zz = np.meshgrid(r, z, indexing='ij')
    # print('rr shape = ' + str(rr.shape))
    # print('zz shape = ' + str(zz.shape))

    n_r = len(r)
    n_z = len(z)

    # define all coils
    loop_radius_list = []
    loop_current_list = []
    z0_list = []
    color_list = []
    scale_list = []

    ## main section ##
    z1 = min(z)
    z2 = max(z)
    # z_solenoid_end = z1 + 0.4 * (z2 - z1)
    z_solenoid_end = 0
    # num_main_cell_coils = 10
    # z_MMM_start = z_solenoid_end + solenoid_r / 2
    # z_MMM_start = z_solenoid_end + solenoid_r / 4
    z_MMM_start = z_solenoid_end
    num_main_cell_coils = int(np.round((z_solenoid_end - z1) / (solenoid_r / 2)))
    # print('num_main_cell_coils = ' + str(num_main_cell_coils))
    loop_radius_list_main_cell = [1 for i in range(num_main_cell_coils)]

    I0 = 1.0
    loop_current_list_main_cell = [I0 for i in range(num_main_cell_coils)]
    z0_list_main_cell = np.ndarray.tolist(np.linspace(z1, z_solenoid_end, num_main_cell_coils))
    color_list_main_cell = ['b' for i in range(num_main_cell_coils)]
    scale_list_main_cell = [0.5 for i in range(num_main_cell_coils)]

    loop_radius_list += loop_radius_list_main_cell
    loop_current_list += loop_current_list_main_cell
    z0_list += z0_list_main_cell
    color_list += color_list_main_cell
    scale_list += scale_list_main_cell

    ### MMM section DC coils ###
    num_MM_coils = int(np.round((z2 - z_MMM_start) / (solenoid_r / 2)))
    # print('num MM DC coils = ' + str(num_MM_coils))
    loop_radius_list_MMM_DC = [1 for i in range(num_MM_coils)]
    color_list_MMM_DC = ['r' for i in range(num_MM_coils)]

    # print('MMM DC_current = ' + str(DC_current))

    loop_current_list_MMM_DC = [DC_current for i in range(num_MM_coils)]
    scale_list_MMM_DC = np.ndarray.tolist(0.5 * np.array(loop_current_list_MMM_DC) / max(loop_current_list_MMM_DC))
    z_MMM_array = np.linspace(z_MMM_start, z2, num_MM_coils)
    z0_list_MMM_DC = np.ndarray.tolist(z_MMM_array)

    loop_radius_list += loop_radius_list_MMM_DC
    loop_current_list += loop_current_list_MMM_DC
    z0_list += z0_list_MMM_DC
    color_list += color_list_MMM_DC
    scale_list += scale_list_MMM_DC

    ### MMM section AC coils ###
    coils_per_wavelength = 2
    num_MM_coils = int(np.round((z2 - z_MMM_start) / lambda_MMM) * coils_per_wavelength)
    # print('num MM AC coils = ' + str(num_MM_coils))
    loop_radius_list_MMM_AC = [1 for i in range(num_MM_coils)]
    color_list_MMM_AC = ['r' for i in range(num_MM_coils)]
    z_MMM_array = np.linspace(z_MMM_start, z2, num_MM_coils)
    phase = np.pi / 2

    # print('MMM AC_current = ' + str(AC_current))
    # print('MMM negative_AC_current = ' + str(negative_AC_current))

    loop_current_list_MMM_AC = z_MMM_array * 0
    for i in range(num_MM_coils):
        if np.mod(i, 2) == 0:
            loop_current_list_MMM_AC[i] = AC_current
        else:
            loop_current_list_MMM_AC[i] = - negative_AC_current

    scale_list_MMM_AC = np.ndarray.tolist(0.5 * loop_current_list_MMM_AC / np.max(loop_current_list_MMM_AC))
    loop_current_list_MMM_AC = np.ndarray.tolist(loop_current_list_MMM_AC)
    z0_list_MMM_AC = np.ndarray.tolist(z_MMM_array)

    loop_radius_list += loop_radius_list_MMM_AC
    loop_current_list += loop_current_list_MMM_AC
    z0_list += z0_list_MMM_AC
    color_list += color_list_MMM_AC
    scale_list += scale_list_MMM_AC

    ### Plots ###

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

    label = '$I_{DC}$ = ' + str(DC_current / I0) \
            + ', $I_{AC}$ = ' + str(AC_current / I0) \
            + ', $I_{negAC}$ = ' + str(negative_AC_current / I0)
    title_suffix = ', $\\lambda_{MMM}/r=$' + str(lambda_MMM / solenoid_r)

    # plot current as function of position on axis
    if plot_curves is True:
        plt.figure(fig_num)
        plt.plot(z0_list, loop_current_list, 'o',
                 # label='lambda_MMM / r = ' + str(lambda_MMM / solenoid_r) + ', pos_frac = ' + str(pos_frac))
                 # label='lambda_MMM / r = ' + str(lambda_MMM / solenoid_r))
                 # label='$I_{DC}$ = ' + str(DC_current / I0) + ', $I_{AC}$ = ' + str(AC_current / I0))
                 label=label)
        plt.xlabel('z [m]')
        plt.ylabel('Current')
        plt.title('Current in coils' + title_suffix)
        # plt.tight_layout()
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
    # print('Rm = ' + str(Rm))
    # Rm_list[ind_lambda] = Rm

    # plot magnetic field profile on axis
    B0 = B_z_tot[int(B_z_tot.shape[0] / 2), int(B_z_tot.shape[1] / 8)]
    if plot_curves is True:
        plt.figure(fig_num + 1)
        # plt.plot(z, B_z_tot[int(B_z_tot.shape[0]/2), :], label='MMM phase = ' + str(phase / np.pi) + '$\pi$')
        plt.plot(z,
                 B_z_tot[int(B_z_tot.shape[0] / 2), :] / B0,
                 # label='lambda_MMM / r = ' + str(lambda_MMM / solenoid_r))
                 # label='$I_{DC}$ = ' + str(DC_current / I0) + ', $I_{AC}$ = ' + str(AC_current / I0))
                 label=label)
        # label = 'lambda_MMM / r = ' + str(lambda_MMM / solenoid_r) + ', pos_frac = ' + str(pos_frac) + ', Rm =' + str(Rm))

        # label = 'MMM phase = ' + str(phase / np.pi) + '$\pi$')
        plt.xlabel('z [m]')
        plt.ylabel('$B_z/B_{z,0}$')
        plt.title('Axial magnetic field on axis (normalized)' + title_suffix, size=15)
        # plt.tight_layout()
        plt.legend()
        plt.grid(True)

    return B0, B_max, B_min, Rm


###########################################

# plt.close('all')

# linestyle = '-'
linestyle = '--'
# linestyle = ':'

# lambda_MMM_array = solenoid_r * np.linspace(3, 10, 8)
# lambda_MMM_array = solenoid_r * np.array([3, 4, 5, 6, 8, 10])
lambda_MMM_array = solenoid_r * np.array([3, 4, 5, 6, 10])
# lambda_MMM_array = solenoid_r * np.array([4])
# lambda_MMM_array = solenoid_r * np.array([10])


for ind_lambda, lambda_MMM in enumerate(lambda_MMM_array):
    print('ind_lambda = ' + str(ind_lambda))

    # AC_current_array = I0 * np.linspace(1, 10, 5)
    AC_current_array = I0 * np.linspace(1, 15, 15)
    # AC_current_array = I0 * np.array([5])
    # AC_current_array = I0 * np.array([10])

    Rm_only_AC_array = 0 * AC_current_array
    Rm_AC_and_DC_array = 0 * AC_current_array
    required_DC_array = 0 * AC_current_array

    for ind_AC, AC_current in enumerate(AC_current_array):
        DC_current_0 = 0
        AC_current_0 = AC_current
        # negative_AC_current_0 = 0
        # negative_AC_current_0 = 0.5 * AC_current
        negative_AC_current_0 = 1 * AC_current

        # label = '$\\lambda_{MMM}/r=$' + str(lambda_MMM / solenoid_r)
        label = '$\\lambda_{MMM}/r=$' + str(lambda_MMM / solenoid_r) + ' with $I_{negAC}/I_{AC}=$' + str(
            negative_AC_current_0 / AC_current_0)

        B0, B_max_0, B_min_0, Rm_0 = calc_mirror_ratio(lambda_MMM, DC_current=DC_current_0, AC_current=AC_current_0,
                                                       negative_AC_current=negative_AC_current_0)

        # AC_current = AC_current_0 * B0 / B_min_0
        # DC_current = 0
        # B0, B_max, B_min, Rm = calc_mirror_ratio(lambda_MMM, DC_current=DC_current, AC_current=AC_current,
        #                                          negative_AC_current=negative_AC_current_0)
        # Rm_only_AC_array[ind_AC] = Rm

        DC_current = I0 * (B0 - B_min_0) / B0
        required_DC_array[ind_AC] = DC_current
        B0, B_max, B_min, Rm = calc_mirror_ratio(lambda_MMM, DC_current=DC_current, AC_current=AC_current_0,
                                                 negative_AC_current=negative_AC_current_0)
        Rm_AC_and_DC_array[ind_AC] = Rm

        # DC_current_0 = 0
        # AC_current_0 = 5 * I0
        # negative_AC_current_0 = 5 * I0
        # B0, B_max_0, B_min_0, Rm_0 = calc_mirror_ratio(lambda_MMM, DC_current=DC_current_0, AC_current=AC_current_0,
        #                                                negative_AC_current=negative_AC_current_0)
        #
        # AC_current = AC_current_0 * np.abs(B0 / B_min_0)
        # negative_AC_current = negative_AC_current_0 * np.abs(B0 / B_min_0)
        # DC_current = 0
        # B0, B_max, B_min, Rm = calc_mirror_ratio(lambda_MMM, DC_current=DC_current, AC_current=AC_current,
        #                                          negative_AC_current=negative_AC_current)
        #
        # DC_current = I0 * (B0 - B_min_0) / B0
        # B0, B_max, B_min, Rm = calc_mirror_ratio(lambda_MMM, DC_current=DC_current, AC_current=AC_current_0,
        #                                          negative_AC_current=negative_AC_current_0)

    plt.figure(1)
    plt.plot(AC_current_array / I0, required_DC_array, '-o', linewidth=2, linestyle=linestyle, label=label)
    plt.xlabel('$I_{AC} / I_0$', size=20)
    plt.ylabel('$I_{DC} / I_0$', size=20)
    plt.title('Required DC current for minimal field constraint', size=20)
    # plt.tight_layout()
    plt.legend()
    plt.grid(True)

    plt.figure(2)
    plt.plot(AC_current_array / I0, Rm_AC_and_DC_array, '-o', linewidth=2, linestyle=linestyle, label=label)
    plt.xlabel('$I_{AC} / I_0$', size=20)
    plt.ylabel('$R_m$', size=20)
    plt.title('Mirror Ratio $B_{max}/B_{min}$ with AC and DC currents', size=20)
    # plt.tight_layout()
    plt.legend()
    plt.grid(True)
