import matplotlib.pyplot as plt


def plot_loop_symbols(z0=0, loop_radius=1.0, scale=1, color='k',
                      length_units='m'):
    if length_units == 'cm':
        z0 = z0 * 1e2
        loop_radius = loop_radius * 1e2

    # plot loop location major circles
    plt.plot(z0, loop_radius, 'o', fillstyle='none',
             linewidth=3 * scale, markersize=20 * scale,
             color=color, markeredgewidth=3 * scale)
    plt.plot(z0, -loop_radius, 'o', fillstyle='none',
             linewidth=3 * scale, markersize=20 * scale,
             color=color, markeredgewidth=3 * scale)
    # plot loop current direction
    plt.plot(z0, loop_radius, 'o', fillstyle='full',
             linewidth=3 * scale, markersize=7 * scale,
             color=color, markeredgewidth=1 * scale)
    plt.plot(z0, -loop_radius, 'x', fillstyle='none',
             linewidth=3 * scale, markersize=9 * scale,
             color=color, markeredgewidth=3 * scale)
    return


def plot_magnetic_field_lines(z, y, B_z, B_y, xlabel='z', ylabel='r',
                              title='$(B_z, B_r)$ vector field',
                              start_points=None, color='k', length_units='m'):
    if length_units == 'cm':
        z = z * 1e2
        y = y * 1e2

    if start_points is not None:
        plt.streamplot(z, y, B_z, B_y, color=color, density=5, linewidth=1,
                       start_points=start_points)
    else:
        plt.streamplot(z, y, B_z, B_y, color=color, density=2, linewidth=1)

    plt.show()
    if xlabel is not None:
        plt.xlabel(xlabel + ' [' + length_units + ']', size=15)
    if ylabel is not None:
        plt.ylabel(ylabel + ' [' + length_units + ']', size=15)
    if title is not None:
        plt.title(title, size=15)
    return
