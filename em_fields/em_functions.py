import numpy as np
from scipy.linalg import expm


def get_thermal_velocity(T, m, kB_eV):
    '''
    The thermal velocity of particles according to 1/2*m*v_th^2 = 3/2*kB*T would give v_th=sqrt(3kBT/m), as we use in
    the mm rate eqs model. But here we want to use the Maxwell Boltzmann most probable velocity which has sqrt(2)
    T in eV, m in kg
    '''
    return np.sqrt(2 * kB_eV * T / m)


def get_cyclotron_angular_frequency(q, B, m):
    '''
    Calculate the angular frequency of cyclotron / Larmor precession (of ion on electron, depending on m)
    q in Coulomb, B in Tesla, m in kg
    '''
    return np.abs(q * B / m)


def get_plasma_frequency(ne, qe, me, eps0):
    '''
    Calculate the angular frequency of plasma oscillations / Langmuir waves of the electrons
    qe in Coulomb, me in kg, ne the number density of electrons in m^-3, eps0 is the vacuum permittivity [Farad/m^2]
    Sources: https://en.wikipedia.org/wiki/Plasma_oscillation, Nicholson
    '''
    return np.sqrt(qe ** 2 * ne / me / eps0)


def evolve_particle_in_em_fields(x_0, v_0, dt, E_function, B_function, field_dict=None, t_0=0, q=1.0, m=1.0,
                                 stop_criterion='steps', num_steps=None, t_max=None, return_fields=True,
                                 number_of_cell_center_crosses=None):
    """
    Advance a charged particle in time under the influence of E,B fields.
    """
    if stop_criterion == 'time':
        num_steps = t_max / dt
    elif stop_criterion == 'first_cell_center_crossing':
        num_steps = int(1e15)  # picking an "infinite" number
    elif stop_criterion == 'several_cell_center_crossing':
        num_steps = int(1e15)  # picking an "infinite" number
        cnt_cell_center_crosses = 1
        inds_cell_center_crossing = [0]
    t = t_0

    if field_dict is None:
        field_dict = {}

    # define a dictionary that will collect all the particles history as it goes
    hist = {}
    hist['x'] = [x_0]
    hist['v'] = [v_0]
    hist['t'] = [t_0]
    if return_fields is True:
        hist['E'] = [E_function(x_0, t_0, **field_dict)]
        hist['B'] = [B_function(x_0, t_0, **field_dict)]

    for ind_step in range(num_steps):
        x_new, v_new = particle_integration_step(hist['x'][-1], hist['v'][-1], hist['t'][-1],
                                                 dt, E_function, B_function, q=q, m=m, field_dict=field_dict)
        hist['x'] += [x_new]
        hist['v'] += [v_new]
        t += dt
        hist['t'] += [t]

        if return_fields is True:
            hist['E'] += [E_function(x_new, t, **field_dict)]
            hist['B'] += [B_function(x_new, t, **field_dict)]

        if stop_criterion in ['first_cell_center_crossing', 'several_cell_center_crossing']:
            z_curr = hist['x'][-1][2]
            z_last = hist['x'][-2][2]

            if abs(np.mod(z_last / field_dict['l'], 1) - np.mod(z_curr / field_dict['l'], 1)) < 0.1:
                # check that during the time step the particle crossed the center of a cell
                if abs(np.mod(z_last / field_dict['l'] - 0.5, 1) - np.mod(z_curr / field_dict['l'] - 0.5, 1)) > 0.5:
                    # avoid getting it straight in the beginning of the simulation
                    if ind_step >= 5:
                        if stop_criterion == 'first_cell_center_crossing':
                            break
                        elif stop_criterion == 'several_cell_center_crossing':
                            inds_cell_center_crossing += [ind_step + 1]
                            cnt_cell_center_crosses += 1
                            if cnt_cell_center_crosses == number_of_cell_center_crosses:
                                break

    for key in hist:
        hist[key] = np.array(hist[key])
    if stop_criterion == 'several_cell_center_crossing':
        hist['inds_cell_center_crossing'] = inds_cell_center_crossing

    return hist


def particle_integration_step(x_0, v_0, t, dt, E_function, B_function, q=1.0, m=1.0, field_dict=None):
    """
    Algorithm based on "2015 - He et al - Volume-preserving algorithms for charged particle dynamics"
    https://www.sciencedirect.com/science/article/pii/S0021999114007141
    """
    x_half = x_0 + dt * v_0 / 2.0
    t_half = t + dt / 2.0
    E_half = E_function(x_half, t_half, **field_dict)
    # E_half = E_function(x_half, t, **field_dict)
    v_minus = v_0 + dt * q / m / 2.0 * E_half
    B_half = B_function(x_half, t_half, **field_dict)
    # B_half = B_function(x_half, t, **field_dict)
    B_norm = np.linalg.norm(B_half)
    b_x = B_half[0] / B_norm
    b_y = B_half[1] / B_norm
    b_z = B_half[2] / B_norm
    b_half_tensor = np.array([[0, -b_z, b_y], [b_z, 0, -b_x], [-b_y, b_x, 0]])
    # omega_half = - q * B_norm / m # definition with minus from paper is inconsistent with "right hand rule"
    omega_half = q * B_norm / m
    v_plus = np.dot(expm(dt * omega_half * b_half_tensor), v_minus)
    v_new = v_plus + dt * q / m / 2.0 * E_half
    x_new = x_half + dt / 2.0 * v_new
    return x_new, v_new
