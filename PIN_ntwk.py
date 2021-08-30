"""
Class implementing PINning networks as in
Rajan (2016): Recurrent networks for sequential memory


Created on 8/25/2021 by 
@email: vyomr@uw.edu
"""

import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


class PIN_ntwk:
    """Network of rate-based recurrently connected neurons with RLS learning"""

    def __init__(self, N_neurons, tau_neuron, duration, dt, n_timesteps,
                 g_wt_variance, theta, seed=0):
        self.N_neurons = N_neurons
        self.tau_neuron = tau_neuron
        self.duration = duration
        self.dt = dt
        self.n_timesteps = n_timesteps
        self.g_wt_variance = g_wt_variance
        self.theta = theta
        self.rng = np.random.default_rng(seed)
        self.external_input = None
        self.arr_P_init = None
        self.arr_J_init = None
        self.p_plastic = None
        self.n_plastic = None
        self.rng_train = None
        self.arr_plastic_neurons = None
        self.alpha = None
        self.arr_J_learned = None
        self.arr_P_learned = None
        self.arr_dJ_mag = None
        self.dict_arr_J_learned = {}

    def generate_frozen_external_input(self, tau_wn, h0_wn):
        sqrt_dt = np.sqrt(self.dt)
        sqrt_s = np.sqrt(b2.second)  # Not sure about this implementation, might be getting lucky since tau_wn = 1s
        external_input = np.zeros((self.N_neurons, self.n_timesteps))

        for idx_t in range(1, self.n_timesteps):
            h = external_input[:, idx_t - 1]
            external_input[:, idx_t] = h + self.dt * (-h / tau_wn) + \
                                       h0_wn * self.rng.normal(size=self.N_neurons) * sqrt_dt / sqrt_s

        self.external_input = external_input

    def generate_weights(self):
        # Initialize random weight matrix
        self.arr_J_init = self.rng.normal(0, np.sqrt(self.g_wt_variance ** 2 / self.N_neurons),
                                          size=(self.N_neurons, self.N_neurons))

    # TODO refactor train and run to avoid duplication
    def train(self, arr_target_fx, p_plastic, alpha=1, n_training_steps=21, n_plot_every=5,
              rule='syntot', seed=0, b_inject_noise=False, amp_noise=0.1):
        # Select plastic neurons
        self.p_plastic = p_plastic
        self.n_plastic = int(p_plastic * self.N_neurons)
        self.rng_train = np.random.default_rng(seed)
        self.arr_plastic_neurons = self.rng_train.choice(np.arange(self.N_neurons), self.n_plastic, replace=False)

        # Initialize arr_P
        self.alpha = alpha
        self.arr_P_init = (1.0 / self.alpha) * np.eye(self.N_neurons)[:, self.arr_plastic_neurons]

        neuron_output_x = np.zeros((self.N_neurons, self.n_timesteps))
        neuron_output_rate = np.zeros((self.N_neurons, self.n_timesteps))
        neuron_output_syntot = np.zeros((self.N_neurons, self.n_timesteps))
        neuron_error = np.zeros((self.N_neurons, self.n_timesteps))
        self.arr_dJ_mag = np.zeros((n_training_steps, self.n_timesteps))

        # Select error rule
        if rule == 'syntot':
            arr_optimize = neuron_output_syntot
        elif rule == 'x':
            arr_optimize = neuron_output_x
        elif rule == 'rate':
            arr_optimize = neuron_output_rate

        arr_J = self.arr_J_init.copy()
        arr_P = self.arr_P_init.copy()

        # Start figure
        n_fig_rows = int(n_training_steps / n_plot_every)
        if n_plot_every > 1:
            n_fig_rows += 1
        idx_fig_row = 0
        f, axs = plt.subplots(figsize=(6, 3 * n_fig_rows), ncols=2, nrows=n_fig_rows)
        axs = axs.reshape(n_fig_rows, 2)
        axs[0, 1].set_title('dJ')
        for idx_learn in tqdm(range(n_training_steps)):
            # reset x at t0
            # neuron_output_x[:, 0] = neuron_output_x[:, -1]
            for idx_t in range(1, self.n_timesteps):
                x = neuron_output_x[:, idx_t - 1]

                neuron_output_rate[:, idx_t] = 1 / (1 + np.exp(-(x - self.theta)))
                arr_r = neuron_output_rate[:, idx_t].reshape(self.N_neurons, 1)

                neuron_output_syntot[:, idx_t] = (arr_J @ arr_r).reshape(self.N_neurons)

                neuron_output_x[:, idx_t] = x + self.dt * (-x + self.external_input[:, idx_t]
                                                           + neuron_output_syntot[:, idx_t]) / self.tau_neuron

                # Inject noise
                if b_inject_noise:
                    neuron_output_x[:, idx_t] += self.dt * amp_noise * self.rng_train.normal(self.N_neurons)/self.tau_neuron

                neuron_error[:, idx_t] = arr_optimize[:, idx_t] - arr_target_fx[:, idx_t]
                arr_e = neuron_error[:, idx_t].reshape(self.N_neurons, 1)

                # wt learning
                arr_r = arr_r[self.arr_plastic_neurons]
                arr_Pr = (arr_P @ arr_r)  # NxN_p@N_px1 = Nx1
                n_rPr = (arr_r.T @ arr_Pr[self.arr_plastic_neurons])[0, 0]  # 1xN_p@N_px1 = 1x1 scalar
                n_c = 1.0 / (1.0 + n_rPr)  # scalar

                # TODO understand this algorithm
                # Update P and J
                #arr_P -= n_c * (arr_Pr @ arr_r.T @ arr_P[self.arr_plastic_neurons, :])  # c * Nx1@1xN_p@N_pxN_p = NxN_p
                arr_P -= n_c * (arr_Pr @ arr_Pr[self.arr_plastic_neurons].T)  # c * Nx1@1xN_p = NxN_p
                arr_dJ = n_c * (arr_e @ arr_Pr[self.arr_plastic_neurons].T)  # c * Nx1@1xN_p = NxN_p
                arr_J[:, self.arr_plastic_neurons] -= arr_dJ
                self.arr_dJ_mag[idx_learn, idx_t] = np.sum(np.abs(arr_J - self.arr_J_init))/np.sum(np.abs(arr_J))

            if (idx_learn % n_plot_every) == 0:
                mappable = axs[idx_fig_row, 0].matshow(neuron_output_rate, aspect='auto', vmax=1,
                                                       vmin=0)  # , cmap='jet')
                plt.colorbar(mappable, ax=axs[idx_fig_row, 0])
                axs[idx_fig_row, 0].set_ylabel('trial {}'.format(idx_learn), size='large')

                mappable = axs[idx_fig_row, 1].matshow(arr_dJ, cmap='bwr', aspect='auto')
                axs[idx_fig_row, 1].set_yticks([])
                plt.colorbar(mappable, ax=axs[idx_fig_row, 1])
                idx_fig_row += 1

        self.arr_J_learned = arr_J
        self.dict_arr_J_learned['rule_{}_p_{}'.format(rule, str(p_plastic))] = arr_J
        self.arr_P_learned = arr_P


    def plot_band_structure(self, str_save=None):
        f, axs = plt.subplots(figsize=(10, 5), ncols=2, gridspec_kw={'width_ratios': [3, 7]})
        # Plot learned weights and band structure
        mappable = axs[0].imshow(self.arr_J_learned, cmap='bwr')
        plt.colorbar(mappable, ax=axs[0])

        df_bands = self.band_structure()
        axs[1].set_xlabel('i - j', fontweight='bold')
        axs[1].axvline(x=0, lw=1, c='k')
        axs[1].axhline(y=0, lw=1, c='k', ls='--')
        axs[1].errorbar(df_bands.loc[:, 'i_j'], df_bands.loc[:, 'mean'],
                        yerr=df_bands.loc[:, 'sd'], ecolor='C3', alpha=0.4,
                        marker='o', c='C3', ms=4)
        axs[1].yaxis.tick_right()

        if str_save:
            plt.savefig(str_save, bbox_inches='tight')

    def run(self, arr_target_fx, rule='syntot', b_inject_noise=False, amp_noise=0.1):
        """
        returns neuron_output_x, neuron_output_rate, neuron_output_syntot, neuron_error
        """
        neuron_output_x = np.zeros((self.N_neurons, self.n_timesteps))
        neuron_output_rate = np.zeros((self.N_neurons, self.n_timesteps))
        neuron_output_syntot = np.zeros((self.N_neurons, self.n_timesteps))
        neuron_error = np.zeros((self.N_neurons, self.n_timesteps))

        # Select error rule
        if rule == 'syntot':
            arr_optimize = neuron_output_syntot
        elif rule == 'x':
            arr_optimize = neuron_output_x
        elif rule == 'rate':
            arr_optimize = neuron_output_rate

        for idx_t in range(1, self.n_timesteps):
            x = neuron_output_x[:, idx_t - 1]

            neuron_output_rate[:, idx_t] = 1 / (1 + np.exp(-(x - self.theta)))
            arr_r = neuron_output_rate[:, idx_t].reshape(self.N_neurons, 1)

            neuron_output_syntot[:, idx_t] = (self.arr_J_learned @ arr_r).reshape(self.N_neurons)

            neuron_output_x[:, idx_t] = x + self.dt * (-x + self.external_input[:, idx_t]
                                                       + neuron_output_syntot[:, idx_t]) / self.tau_neuron

            # Inject noise
            if b_inject_noise:
                neuron_output_x[:, idx_t] += self.dt * amp_noise * self.rng_train.normal(self.N_neurons)/self.tau_neuron

            neuron_error[:, idx_t] = arr_optimize[:, idx_t] - arr_target_fx[:, idx_t]
        return neuron_output_x, neuron_output_rate, neuron_output_syntot, neuron_error

    def band_structure(self):
        ls_diag_means = []
        ls_diag_sds = []

        for idx_d in range(-99, 100):
            arr_diag = self.arr_J_learned.diagonal(idx_d)
            ls_diag_means.append(arr_diag.mean())
            ls_diag_sds.append(arr_diag.std())

        df_bands = pd.DataFrame(columns=['i_j', 'mean', 'sd'])
        df_bands.i_j = range(-99, 100)
        df_bands['mean'] = ls_diag_means
        df_bands['sd'] = ls_diag_sds

        return df_bands
