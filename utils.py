"""
Utility functions for network analysis

Created on 8/25/2021 by 
@email: vyomr@uw.edu
"""

import numpy as np
import brian2 as b2


def kth_diag_indices(a, k):
    """
    Get indices of kth diagonal of square array

    :param a: input array
    :param k: diagonal index
    :return: rows, cols
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def get_pvar(arr_target_fx, arr_net_output):
    """
    pVar metric implementation for measuring network performance wrt target functions.
    See Rajan (2016) SI 7.7 for details.
    pVar = 1 - <D_i(t) - r_i(t)>^2/<D_i(t) - D_bar_i(t)>^2
         = 1 - frobenius_norm(data - outputs)/frobenius_norm(data_variance)

    :param arr_target_fx:
    :param arr_net_output:
    :return: pVar scalar metric
    """
    n_error_norm = np.linalg.norm(arr_target_fx - arr_net_output, ord='fro')
    n_var_norm = np.linalg.norm(arr_target_fx - arr_target_fx.mean(axis=0), ord='fro')

    return 1 - (n_error_norm / n_var_norm)


def generate_target_seq(N_neurons, n_timesteps, dt, std):
    """
    Generate translated gaussian sequence

    :param N_neurons:
    :param n_timesteps:
    :param dt:
    :param std:
    :return: arr_rates_ideal, arr_target_fx
    """
    arr_rates_ideal = np.zeros((N_neurons, n_timesteps))
    arr_target_fx = np.zeros((N_neurons, n_timesteps))

    arr_x = np.arange(n_timesteps)
    std_ideal = int(std * b2.second / dt)  # how many dt's

    for idx in range(N_neurons):
        center_ideal = idx * n_timesteps / N_neurons
        arr_rates_ideal[idx, :] = np.exp(-((arr_x - center_ideal) / (2 * std_ideal)) ** 2)
        arr_target_fx[idx, :] = np.log(arr_rates_ideal[idx, :] / (1 - arr_rates_ideal[idx, :] + 0.00001))

    return arr_rates_ideal, arr_target_fx
