import os
num_threads = "8"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import copy
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import autograd.numpy as npa
import sys
import gwot
import importlib

import sys; sys.path += ['../']

import lineageot_files.simulation as sim
import lineageot_files.inference as sim_inf
import lineageot_files.new as sim_new
from lineageot_files.new import RMS
import gwot.bridgesampling as bs

import models_mfl as models

np.random.seed(2)

flow_type = 'MFL_2'

# Setting simulation parameters
timescale = 1
diffusion_constant, division_time_distribution = .15, "exponential"
device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)

# Choosing which of the three dimensions to show in later plots
dimensions_to_plot = [0, 1]

N_list = np.array([3, 5, 7, 10, 15])
n_mean = [50, 40, 30, 20, 10]
factor_gt = 5 # factor providing the number of cells for the ground-truth
M = 100
n_iter = 500

x_init = np.array([0, 0, 0])
T = np.linspace(0.01, 1.1, 7)
number_cells = np.array([M for i in range(0, len(T))], dtype=int)
t_idx_mfl = np.zeros(np.sum(number_cells), dtype=int)
cnt_low, cnt_high = 0, 0
for i in range(0, len(T)):
    cnt_high += number_cells[i]
    t_idx_mfl[cnt_low:cnt_high] = i
    cnt_low = cnt_high


def simulate_groundtruth(N0, flow_type):
    non_constant_branching_rate = False
    mean_division_time = 1e6
    sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                          flow_type = flow_type,
                                          mutation_rate = 1/timescale,
                                          mean_division_time = mean_division_time,
                                          timestep = 0.005*timescale,
                                          initial_distribution_std = 0.1,
                                          diffusion_constant = diffusion_constant,
                                          division_time_distribution = division_time_distribution,
                                          non_constant_branching_rate = non_constant_branching_rate
                                         )

    sample = []
    initial_cell = []
    for cnt, t in enumerate(T):
        tmp = []
        tmp_initial_cell = []
        for n in range(0, N0):
            tmp_initial_cell.append(sim.Cell(np.random.normal(x_init, sim_params.initial_distribution_std),
                                    np.zeros(sim_params.barcode_length)))
            tmp.append(sim.sample_descendants(tmp_initial_cell[n], t, t, sim_params))
        sample.append(tmp)
        initial_cell.append(tmp_initial_cell)
    # Extracting trees and barcode matrices
    true_trees = [[sim_inf.list_tree_to_digraph(sample[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    for cnt, t in enumerate(T):
        for n in range(N0):
            true_trees[cnt][n].nodes['root']['cell'] = initial_cell[cnt][n]

    # Computing the ground-truth coupling

    data_arrays = [[sim_inf.extract_data_arrays(true_trees[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    rna_arrays = []
    for t in range(0, len(T)):
        tmp_array = data_arrays[t][0][0]
        for n in range(1, N0):
            tmp_array = np.concatenate((tmp_array, data_arrays[t][n][0]), axis=0)
        rna_arrays.append(tmp_array)

    return sim_new.build_samples_real(rna_arrays)[0]


def MFL_branching(N0, flow_type):
    non_constant_branching_rate = True
    mean_division_time = .25
    sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                          flow_type = flow_type,
                                          mutation_rate = 1/timescale,
                                          mean_division_time = mean_division_time,
                                          timestep = 0.005*timescale,
                                          initial_distribution_std = 0.1,
                                          diffusion_constant = diffusion_constant,
                                          division_time_distribution = division_time_distribution,
                                          non_constant_branching_rate = non_constant_branching_rate
                                         )


    sample = []
    initial_cell = []
    for cnt, t in enumerate(T):
        tmp = []
        tmp_initial_cell = []
        for n in range(0, N0):
            tmp_initial_cell.append(sim.Cell(np.random.normal(x_init, sim_params.initial_distribution_std),
                                    np.zeros(sim_params.barcode_length)))
            tmp.append(sim.sample_descendants(tmp_initial_cell[n], t, t, sim_params))
        sample.append(tmp)
        initial_cell.append(tmp_initial_cell)
    # Extracting trees and barcode matrices
    true_trees = [[sim_inf.list_tree_to_digraph(sample[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    for cnt, t in enumerate(T):
        for n in range(N0):
            true_trees[cnt][n].nodes['root']['cell'] = initial_cell[cnt][n]

    # Computing the ground-truth coupling

    data_arrays = [[sim_inf.extract_data_arrays(true_trees[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    rna_arrays = []
    for t in range(0, len(T)):
        tmp_array = data_arrays[t][0][0]
        for n in range(1, N0):
            tmp_array = np.concatenate((tmp_array, data_arrays[t][n][0]), axis=0)
        rna_arrays.append(tmp_array)

    samples_real, t_idx_real = sim_new.build_samples_real(rna_arrays)

    previous_rna_array = []
    for t in range(1, len(T)):
        tmp_array = sim_inf.extract_ancestor_data_arrays(true_trees[t][0], T[t-1], sim_params)[0]
        for n in range(1, N0):
            tmp_array = np.concatenate((tmp_array, sim_inf.extract_ancestor_data_arrays(true_trees[t][n], T[t-1], sim_params)[0]), axis=0)
        previous_rna_array.append(tmp_array)

    num_cells = [rna_arrays[t].shape[0] for t in range(0, len(T))]

    # Creating a copy of the true tree
    true_trees_annotated = [[copy.deepcopy(true_trees[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    for n in range(0, N0):
        for t in range(0, len(T)):
            sim_inf.add_node_times_from_division_times(true_trees_annotated[t][n])
            for tt in range(0, t):
                sim_inf.add_nodes_at_time(true_trees_annotated[t][n], T[tt])

    branching_rate_fn_true = lambda x : 1/sim.func_mean_division_time(x, sim_params)

    print(samples_real.shape, t_idx_real.shape)

    m_without_branchingrate = models.TrajLoss([torch.randn(number_cells[i], samples_real.shape[1])*0.1 for i in range(0, len(number_cells))],
                            [torch.full((num_cells[i], ), 1/num_cells[i], device = device) for i in range(0, len(number_cells))],
                            torch.tensor(samples_real, device = device),
                            torch.tensor(t_idx_real, device = device),
                            dt = (T[-1] - T[0])/len(T), tau = sim_params.diffusion_constant, sigma = None, M = number_cells,
                            lamda_reg = torch.tensor(.025*np.ones(len(T))), lamda_cst = 0, sigma_cst = float("Inf"),
                            branching_rate_fn = None,
                            sinkhorn_iters = 250, device = device, warm_start = True)

    models.optimize(m_without_branchingrate, n_iter = n_iter, eta_final = 0.1, tau_final = sim_params.diffusion_constant,
                                   sigma_final = 0.5, N = np.mean(number_cells), temp_init = 1.0, temp_ratio = 1.0,
                                   dim = samples_real.shape[1], tloss = m_without_branchingrate, print_interval = 50)

    m_with_branchingrate = models.TrajLoss([torch.randn(number_cells[i], samples_real.shape[1])*0.1 for i in range(0, len(number_cells))],
                            [torch.full((num_cells[i], ), 1/num_cells[i], device = device) for i in range(0, len(number_cells))],
                            torch.tensor(samples_real, device = device),
                            torch.tensor(t_idx_real, device = device),
                            dt = (T[-1] - T[0])/len(T), tau = sim_params.diffusion_constant, sigma = None, M = number_cells,
                            lamda_reg = torch.tensor(.025*np.ones(len(T))), lamda_cst = 0, sigma_cst = float("Inf"),
                            branching_rate_fn = branching_rate_fn_true,
                            sinkhorn_iters = 250, device = device, warm_start = True, lamda_unbal = None)

    models.optimize(m_with_branchingrate, n_iter = n_iter, eta_final = 0.1, tau_final = sim_params.diffusion_constant,
                                   sigma_final = 0.5, N = np.mean(number_cells), temp_init = 1.0, temp_ratio = 1.0,
                                   dim = samples_real.shape[1], tloss = m_with_branchingrate, print_interval = 50)


    weights_deconvoluted = sim_new.build_weights_deconvoluted(true_trees_annotated, N0)
    print('WEIGHT', [weights_deconvoluted[i].shape for i in range(0, len(number_cells))])

    m_with_treestructure = models.TrajLoss([torch.randn(number_cells[i], samples_real.shape[1])*0.1 for i in range(0, len(number_cells))],
                            [torch.tensor(weights_deconvoluted[i], device=device) for i in range(0, len(number_cells))],
                            torch.tensor(samples_real, device = device),
                            torch.tensor(t_idx_real, device = device),
                            dt = (T[-1] - T[0])/len(T), tau = sim_params.diffusion_constant, sigma = None, M = number_cells,
                            lamda_reg = torch.tensor(.025*np.ones(len(T))), lamda_cst = 0, sigma_cst = float("Inf"),
                            branching_rate_fn = None,
                            sinkhorn_iters = 250, device = device, warm_start = True)

    models.optimize(m_with_treestructure, n_iter = n_iter, eta_final = 0.1, tau_final = sim_params.diffusion_constant,
                                   sigma_final = 0.5, N = np.mean(number_cells), temp_init = 1.0, temp_ratio = 1.0,
                                   dim = samples_real.shape[1], tloss = m_with_treestructure, print_interval = 50)

    return samples_real, t_idx_real, m_without_branchingrate, m_with_branchingrate, m_with_treestructure

groundtruth = simulate_groundtruth(factor_gt * M, flow_type)

res = []
for cnt, n in enumerate(N_list):
    res.append(list())
    for i in range(0, n_mean[cnt]):
        res[cnt].append(MFL_branching(n, flow_type))

fig, ax = plt.subplots(2, 4, figsize=(20, 10))

res_rms_null = [[] for _ in range(len(N_list))]
res_rms_br = [[] for _ in range(len(N_list))]
res_rms_tr = [[] for _ in range(len(N_list))]

rownorm = lambda x: x/x.sum(1).reshape(-1, 1)

for cnt, m in enumerate(res):
    for i in range(0, n_mean[cnt]):
        res_rms_null[cnt].append(0)
        res_rms_tr[cnt].append(0)
        res_rms_br[cnt].append(0)

        a, b, m_null, m_branching, m_trees = m[i]
        # res_rms_null[cnt][i] += RMS(np.concatenate([m.detach().numpy()
        #                     for m in m_null.x], 0)[:, :2], groundtruth[:, :2])
        # res_rms_tr[cnt][i] += RMS(np.concatenate([m.detach().numpy()
        #                     for m in m_trees.x], 0)[:, :2], groundtruth[:, :2])
        # with torch.no_grad():
        #     paths = bs.sample_paths(None, N = M, coord = True, x_all = [mt.detach().numpy() for mt in m_branching.x],
        #                     get_gamma_fn = lambda j : rownorm(m_branching.loss_reg.ot_losses[j].coupling().cpu()),
        #                             num_couplings = len(T)-1)
        # res_rms_br[cnt][i] += RMS(np.concatenate([paths[:, time, :2]
        #                     for time in range(0, paths.shape[1])], 0)[:, :], groundtruth[:, :2])
        for t in range(len(T)):
            res_rms_null[cnt][i] += RMS(m_null.x[t].detach().numpy()[:, :2],
                                                        groundtruth[t*factor_gt*M:(t+1)*factor_gt*M, :2])/len(T)
            res_rms_tr[cnt][i] += RMS(m_trees.x[t].detach().numpy()[:, :2],
                                                        groundtruth[t*factor_gt*M:(t+1)*factor_gt*M, :2])/len(T)
            with torch.no_grad():
                paths = bs.sample_paths(None, N = M, coord = True, x_all = [mt.detach().numpy() for mt in m_branching.x],
                                get_gamma_fn = lambda j : rownorm(m_branching.loss_reg.ot_losses[j].coupling().cpu()),
                                        num_couplings = len(T)-1)
            res_rms_br[cnt][i] += RMS(paths[:, t, :2], groundtruth[t*factor_gt*M:(t+1)*factor_gt*M, :2])/len(T)

    res_rms_br[cnt][0] = np.median(res_rms_br[cnt])
    res_rms_tr[cnt][0] = np.median(res_rms_tr[cnt])
    res_rms_null[cnt][0] = np.median(res_rms_null[cnt])

p0, = ax[0, 2].plot(N_list, [res_rms_null[cnt][0] for cnt in range(len(N_list))], c='orange', linestyle='dashed', marker='x')
p0.set_label('without correction')
p1, = ax[0, 2].plot(N_list, [res_rms_br[cnt][0] for cnt in range(len(N_list))], c='blue', linestyle='dashdot', marker='x')
p1.set_label('branching rate')
p2, = ax[0, 2].plot(N_list, [res_rms_tr[cnt][0] for cnt in range(len(N_list))], c='green', marker='x')
p2.set_label('reweighting')
ax[0,2].set_title("C", weight='bold')
ax[0, 2].legend(loc='upper right', fontsize=12)
ax[0, 2].set_xlabel('number of trees', fontsize=12)
ax[0, 2].set_ylabel('RMS', fontsize=12)

samples_real, t_idx_real, m_null, m_branching, m_trees = res[int(min(2, len(N_list)-1))][8]

with torch.no_grad():
    paths = bs.sample_paths(None, N = M, coord = True, x_all = [mt.detach().numpy() for mt in m_branching.x],
                            get_gamma_fn = lambda i : rownorm(m_branching.loss_reg.ot_losses[i].coupling().cpu()),
                                    num_couplings = len(T)-1)
for time in range(0, paths.shape[1]):
    paths[:, time, :] += np.random.normal(0, diffusion_constant/len(T), (paths.shape[0], paths.shape[2])) # Add a small noise for visualization only
m_branching = np.concatenate([paths[:, m, :] for m in range(0, paths.shape[1])], 0)
m_trees = np.concatenate([m.detach().numpy() for m in m_trees.x], 0)
m_null = np.concatenate([m.detach().numpy() for m in m_null.x], 0)

with torch.no_grad():
    ax[0,0].scatter(samples_real[:, dimensions_to_plot[0]], samples_real[:, dimensions_to_plot[1]],
                c= t_idx_real, alpha = 0.5)
    ax[0,0].set_title("A", weight='bold')
    ax[0,0].set_xlabel('gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)
    ax[0,0].set_ylabel('gene ' + str(dimensions_to_plot[1] + 1), fontsize=12)
    if flow_type == 'MFL_2':
        ax[0,0].set_xlim(-2, 2)
        ax[0,0].set_ylim(-1.25, .4)

    if flow_type == 'MFL_2':
        ax[1,0].set_xlim(-2, 2)
        ax[1,0].set_ylim(-1.25, .4)
    s = ax[1,0].scatter(m_null[:, dimensions_to_plot[0]],
                m_null[:, dimensions_to_plot[1]],
                c = t_idx_mfl, alpha = 0.5)
    ax[1,0].set_title("D", weight='bold')
    ax[1,0].set_xlabel('gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)
    ax[1,0].set_ylabel('gene ' + str(dimensions_to_plot[1] + 1), fontsize=12)

    ax[1,1].scatter(m_branching[:, dimensions_to_plot[0]], m_branching[:, dimensions_to_plot[1]],
                c = t_idx_mfl, alpha = 0.5)
    ax[1,1].set_title("E", weight='bold')
    if flow_type == 'MFL_2':
        ax[1,1].set_xlim(-2, 2)
        ax[1,1].set_ylim(-1.25, .4)
    ax[1,1].set_xlabel('gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)

    if flow_type == 'MFL_2':
        ax[1,2].set_xlim(-2, 2)
        ax[1,2].set_ylim(-1.25, .4)
    s = ax[1,2].scatter(m_trees[:, dimensions_to_plot[0]],
                m_trees[:, dimensions_to_plot[1]],
                c = t_idx_mfl, alpha = 0.5)
    ax[1,2].set_title("F", weight='bold')
    ax[1,2].set_xlabel('gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)

    if flow_type == 'MFL_2':
        ax[0,1].set_xlim(-2, 2)
        ax[0,1].set_ylim(-1.25, .4)
    ax[0,1].scatter(groundtruth[:, dimensions_to_plot[0]],
                groundtruth[:, dimensions_to_plot[1]],
                c = np.kron(np.arange(len(T)), np.ones(factor_gt*M)), alpha = 0.5)
    ax[0,1].set_title("B", weight='bold')
    ax[0,1].set_xlabel('gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)

    ax[1,3].axis('off')
    cax = plt.axes([0.71, .11, 0.008, 0.33])
    cbar = fig.colorbar(s, ax=ax[1,3], cax=cax)
    cbar.ax.set_title('time', fontsize=12)
    ax[0,3].axis('off')

plt.savefig("Figures/Figure3.pdf".format(flow_type), dpi=150)
plt.close()
