import numpy as np


def define_default_settings(settings=None):
    if settings == None:
        settings = {}

    #### physical constants
    settings['eV'] = 1.0
    settings['keV'] = 1e3 * settings['eV']
    settings['eV_to_K'] = 1.16e4
    settings['MeV_to_J'] = 1e6 * 1.6e-19
    settings['kB_K'] = 1.380649e-23  # J/K
    settings['e'] = 1.60217662e-19  # Coulomb (elementary charge)
    settings['kB_eV'] = settings['kB_K'] * settings['eV_to_K']  # J/eV (numerically same as e)
    settings['eps0'] = 8.85418781e-12  # Farad/m^2 (vacuum permittivity)
    settings['c'] = 3e8  # m/s

    ### plasma parameters5
    if 'gas_name' not in settings:
        settings['gas_name'] = 'hydrogen'
    if 'ionization_level' not in settings:
        settings['ionization_level'] = 1.0
        # settings['ionization_level'] = None
    settings['me'], settings['mp'], settings['mi'], settings['A_atomic_weight'], settings['Z_ion'] \
        = define_plasma_parameters(gas_name=settings['gas_name'], ionization_level=settings['ionization_level'])

    return settings


def define_plasma_parameters(gas_name='hydrogen', ionization_level=1):
    me = 9.10938356e-31  # kg
    mp = 1.67262192e-27  # kg
    if gas_name == 'hydrogen':
        A = 1.00784
        Z = 1.0
    elif gas_name == 'deuterium':
        A = 2.01410177811
        Z = 1.0
    elif gas_name == 'tritium':
        A = 3.0160492
        Z = 1.0
    elif gas_name == 'DT_mix':
        A = np.mean([2.01410177811, 3.0160492])  # approximate as mean of D and T
        Z = 1.0
    elif gas_name == 'helium':
        A = 4.002602
        Z = 2.0
    elif gas_name == 'lithium':
        A = 6.941  # 92.41% Li7 A=7.016, 7.59% Li6 A=6.015 (Wikipedia)
        Z = 3.0
    elif gas_name == 'sodium':
        A = 22.9897
        Z = 11.0
    elif gas_name == 'potassium':
        A = 39.0983
        Z = 19.0
    else:
        raise TypeError('invalid gas: ' + gas_name)
    mi = A * mp
    # for non-fusion experiments with low temperature, the ions are not fully ionized
    if ionization_level is not None:
        Z = ionization_level
    return me, mp, mi, A, Z
