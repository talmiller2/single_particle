import matplotlib.pyplot as plt
import numpy as np


def num_to_coords_str(x, notation_type='both', scilimits=(-1e5, 1e6)):
    if notation_type in ['f', 'float']:
        format_type = 'f'
    elif notation_type in ['e', 'scientific']:
        format_type = 'e'
    else:
        if x < scilimits[0] or x > scilimits[1]:
            format_type = 'e'
        else:
            format_type = 'f'
    return ('{:.4' + format_type + '}').format(x)


def format_coord(x, y, X, Y, Z, **notation_kwargs):
    coords_str = 'x=' + num_to_coords_str(x, **notation_kwargs) + ', y=' + num_to_coords_str(y, **notation_kwargs)
    if len(X.shape) == 1 and len(Y.shape) == 1:
        xarr, yarr = X, Y
    elif len(X.shape) == 2 and len(Y.shape) == 2:
        xarr = X[0, :]
        yarr = Y[:, 0]
    else:
        raise ValueError('invalid X, Y formats.')

    if ((x > xarr.min()) & (x <= xarr.max()) &
            (y > yarr.min()) & (y <= yarr.max())):
        col = np.searchsorted(xarr, x) - 1
        row = np.searchsorted(yarr, y) - 1
        z = Z[row, col]
        coords_str += ', z=' + num_to_coords_str(z, **notation_kwargs) + '  [' + str(row) + ',' + str(col) + ']'
    return coords_str


def update_format_coord(X, Y, Z, ax=None, **notation_kwargs):
    if ax == None:
        ax = plt.gca()
    ax.format_coord = lambda x, y: format_coord(x, y, X, Y, Z, **notation_kwargs)
    plt.show()
