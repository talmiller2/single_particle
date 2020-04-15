import matplotlib.pyplot as plt


def plot_loop_symbols(z0=0, loop_radius=1.0, scale=1):
    # plot loop location major circles
    plt.plot(z0, loop_radius, 'o', fillstyle='none',
             linewidth=3 * scale, markersize=20 * scale,
             color='black', markeredgewidth=3 * scale)
    plt.plot(z0, -loop_radius, 'o', fillstyle='none',
             linewidth=3 * scale, markersize=20 * scale,
             color='black', markeredgewidth=3 * scale)
    # plot loop current direction
    plt.plot(z0, loop_radius, 'o', fillstyle='full',
             linewidth=3 * scale, markersize=7 * scale,
             color='black', markeredgewidth=1 * scale)
    plt.plot(z0, -loop_radius, 'x', fillstyle='none',
             linewidth=3 * scale, markersize=9 * scale,
             color='black', markeredgewidth=3 * scale)
    return


def plot_magnetic_field_lines(z, y, B_z, B_y, xlabel='z [m]', ylabel='y [m]', title='$(B_z, B_y)$ vector field',
                              start_points=None):
    if start_points is not None:
        plt.streamplot(z, y, B_z, B_y, color='b', density=5, linewidth=1, start_points=start_points)
    else:
        plt.streamplot(z, y, B_z, B_y, color='b', density=2, linewidth=1)

    plt.show()
    if xlabel is not None:
        plt.xlabel(xlabel, size=15)
    if ylabel is not None:
        plt.ylabel(ylabel, size=15)
    if title is not None:
        plt.title(title, size=15)
    return
