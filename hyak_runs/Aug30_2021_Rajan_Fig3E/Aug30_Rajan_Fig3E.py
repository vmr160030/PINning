# # Trying to replicate Figure 3E
# Of Rajan (2016)--i.e. trying to train 500 neuron network with fully plastic synapses resulting in oscillating band structure
# 
# Monday, 30 August 2021
# Vyom Raval


import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('/usr/lusers/vyomr/Fairhall_code/PINning/')

import PIN_ntwk as pn
from utils import kth_diag_indices, generate_target_seq


# ## Generate target sequence and fxs
N_neurons = 500
duration = 10.5*b2.second
dt = 1*b2.ms
n_timesteps = int(duration/dt)

arr_rates_ideal, arr_target_fx = generate_target_seq(N_neurons, n_timesteps, dt, std=0.3)

# ## Train
tau_neuron = 10*b2.ms
c_net = pn.PIN_ntwk(N_neurons, tau_neuron, duration, dt, n_timesteps, 
                 g_wt_variance=1.35, theta=0, seed=0)

c_net.generate_frozen_external_input(tau_wn=1*b2.second, h0_wn=1)
c_net.generate_weights()

print('Training . . .')
c_net.train(arr_target_fx=arr_rates_ideal, p_plastic=1, alpha=1, n_training_steps=500, n_plot_every=50,
            rule='rate', seed=0)
plt.savefig('/usr/lusers/vyomr/Fairhall_code/PINning/hyak_runs/Aug30_2021_Rajan_Fig3E/training.png', bbox_inches='tight')
print('Done!')

# Save learned weights
np.savetxt('/usr/lusers/vyomr/Fairhall_code/PINning/hyak_runs/Aug30_2021_Rajan_Fig3E/arr_J_learned.txt',
 c_net.arr_J_learned)