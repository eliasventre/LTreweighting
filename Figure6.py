import os
num_threads = "8"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import ot
import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

import sys; sys.path += ['../.']

from lineageot_files import simulation as sim
from lineageot_files import inference as sim_inf
from lineageot_files import new as sim_new
from lineageot_files.new import RMS, subsampled_tree

import models_mfl_LOT as models_method
import models_mfl as models_basic

np.random.seed(2)

flow_type = 'MFL_4' # flow type used for the simulations
factor_MFL4 = 4
coserr = 1

# Parameters for MFL algorithm
device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)

# Setting simulation parameters
timescale = 1
diffusion_constant, division_time_distribution = factor_MFL4*.2, "exponential"
mean_division_time = .35*timescale
mean_death_time = 1*timescale
x_init = np.array([0, 0.15, 0])
dim = 3

# Choosing which of the three dimensions to show in later plots
dimensions_to_plot = [0, 1]
T = np.linspace(.1, 1.5, 7) # timepoints for simulation
M = 100
n_iter = int(500)
n_sim = 5
number_cells = np.array([M for i in range(0, len(T))], dtype=int)

n_mean = 7 # number of repetitions
p_vec = [1, .8, .6, .3, .15, .05, .01] # sequence of subsampling rates
N0 = 10 # number of simulated trees at each timepoint

chain = 9 # reverse of entropy regularization of the Wasserstein distance used in the chaining
          # in the MFL-like method described in Appendix B2



def simulate_groundtruth(M, mu_0, flow_type, t0, T, weights=[]):
    """Function simulating cells from a distribution mu_0 with weights=weights M cells at a sequence of timepoints T under a SDE without branching and drift = flow_type """
    N0 = np.size(mu_0, 0)
    if len(weights):
        mu_0 = mu_0[np.random.choice(N0, M, p=weights), :]
        N0 = M
    non_constant_branching_rate = False
    mean_division_time = 1e6
    sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                          flow_type = flow_type,
                                          mutation_rate = 1/timescale,
                                          mean_division_time = mean_division_time,
                                          timestep = 0.001*timescale,
                                          initial_distribution_std = 0.1,
                                          diffusion_constant = diffusion_constant,
                                          division_time_distribution = division_time_distribution,
                                          non_constant_branching_rate = non_constant_branching_rate
                                         )

    sample = [[] for _ in range(len(T))]
    for cnt, t in enumerate(T):
        for n in range(0, N0):
            sample[cnt].append(sim.evolve_x_fromx0(mu_0[n, :], t0, t, sim_params))

    # Computing the ground-truth coupling

    data_arrays = [[sample[t][n] for n in range(0, N0)] for t in range(0, len(T))]
    rna_arrays = []
    for t in range(0, len(T)):
        tmp_array = np.array([data_arrays[t][0]])
        for n in range(1, N0):
            tmp_array = np.concatenate((tmp_array, np.array([data_arrays[t][n]])), axis=0)
        rna_arrays.append(tmp_array)
    return rna_arrays



def simulate_trees(N0, T, flow_type, proba_sub):
    """Function simulating N0 trees at a sequence of timepoints T under a branchgin SDE, drift = flow_type
     and branching rates associated to the same flow_type """
    non_constant_branching_rate = True # False if the branching rate is constant
    sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                        flow_type = flow_type,
                                        mutation_rate = 1/timescale,
                                        mean_division_time = mean_division_time,
                                        timestep = 0.001*timescale,
                                        initial_distribution_std = 0.1,
                                        diffusion_constant = diffusion_constant,
                                        division_time_distribution = division_time_distribution,
                                        non_constant_branching_rate = non_constant_branching_rate,
                                        )
    # Simulating a set of samples
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
    true_trees = [[] for t in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            true_trees[t].append(sim_inf.list_tree_to_digraph(sample[t][n]))

    # data all
    true_trees_annotated = [[copy.deepcopy(true_trees[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    for n in range(0, N0):
        for t in range(0, len(T)):
            sim_inf.add_node_times_from_division_times(true_trees_annotated[t][n])
            for tt in range(0, t):
                sim_inf.add_nodes_at_time(true_trees_annotated[t][n], T[tt])

    weights_deconvoluted_all = sim_new.build_weights_deconvoluted(true_trees_annotated, N0)

    # Computing array of cells
    data_arrays = [[] for t in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            data_arrays[t].append(sim_inf.extract_data_arrays(true_trees_annotated[t][n]))
    rna_arrays_all = []
    for t in range(0, len(T)):
        tmp_array = data_arrays[t][0][0]
        for n in range(1, N0):
            tmp_array = np.concatenate((tmp_array, data_arrays[t][n][0]), axis=0)
        rna_arrays_all.append(tmp_array)

    # data subsampled
    empty_list = [[] for _ in range(0, len(T))]
    true_trees_annotated_sub = [[copy.deepcopy(true_trees[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            sim_inf.add_node_times_from_division_times(true_trees_annotated_sub[t][n])
            true_trees_annotated_sub[t][n], empty = subsampled_tree(true_trees_annotated_sub[t][n], proba_sub[t])
            empty_list[t].append(empty)
            for tt in range(0, t+1):
                sim_inf.add_nodes_at_time(true_trees_annotated_sub[t][n], T[tt])

    weights_deconvoluted_sub = sim_new.build_weights_deconvoluted_subsampled(true_trees_annotated_sub, N0, empty_list)
    weights_deconvoluted_sub_ancestors = sim_new.build_weights_deconvoluted_ancestors(true_trees_annotated_sub, T, N0, empty_list)

    # Computing array of cells
    data_arrays_sub = [[] for _ in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            data_arrays_sub[t].append(sim_inf.extract_data_arrays(true_trees_annotated_sub[t][n]))
    rna_arrays_sub = []
    for t in range(0, len(T)):
        tmp_array = data_arrays_sub[t][0][0]
        for n in range(1, N0):
            if not empty_list[t][n]:
                tmp_array = np.concatenate((tmp_array, data_arrays_sub[t][n][0]), axis=0)
        rna_arrays_sub.append(tmp_array)

    # Build ancestors for t-1
    ancestor_info = [[], []]
    rna_arrays_sub_ancestors = []
    for t in range(1, len(T)):
        for n in range(0, N0):
            observed_nodes = np.array(sim_inf.get_leaves(true_trees_annotated_sub[t][n], include_root=False))
            sim_inf.add_conditional_means_and_variances(true_trees_annotated_sub[t][n], observed_nodes)
        tmp = sim_new.get_ancestors(true_trees_annotated_sub[t][0], T[t-1])
        tai0 = tmp[0]
        tai1 = tmp[1]
        for n in range(1, N0):
            if not empty_list[t][n]:
                tmp = sim_new.get_ancestors(true_trees_annotated_sub[t][n], T[t-1])
                tai0 = np.concatenate((tai0, tmp[0]), axis=0)
                tai1 = np.concatenate((tai1, tmp[1]), axis=0)
        ancestor_info[0].append(tai0)
        ancestor_info[1].append(tai1)

        if np.size(rna_arrays_sub[t-1], 0) > np.size(ancestor_info[0][t-1],0):
            data1, data2 = rna_arrays_sub[t-1], ancestor_info[0][t-1]
            m1, m2 = -np.log2(weights_deconvoluted_sub[t-1] * N0), -np.log2(weights_deconvoluted_sub_ancestors[t-1] * N0)
            pairwise_dists = ot.dist(data1, data2)
            pairwise_distm = np.zeros((np.size(m1), np.size(m2)))
            for m1_cnt, msub in enumerate(m1):
                for m2_cnt, manc in enumerate(m2):
                    pairwise_distm[m1_cnt, m2_cnt] = abs(msub - manc)
            alpha = np.sum(pairwise_distm) / (np.sum(pairwise_distm) + np.sum(pairwise_dists))
            dist = alpha * pairwise_dists + (1 - alpha) * pairwise_distm
            mu, nu = np.ones_like(pairwise_dists[:, 0]), np.ones_like(pairwise_dists[0, :])
            P = ot.partial.partial_wasserstein(mu, nu, dist, m=np.sum(nu))
            tmp_rna = np.ones((np.size(nu), dim))
            for j in range(0, np.size(nu)):
                for d in range(0, dim):
                    tmp_rna[j, d] = np.sum(P[:, j] * data1[:, d])
            rna_arrays_sub_ancestors.append(tmp_rna)
        else:
            weights_deconvoluted_sub_ancestors[t-1] = weights_deconvoluted_sub[t-1]
            rna_arrays_sub_ancestors.append(rna_arrays_sub[t-1])

    # MFL algorithm on the datasets with and without subsampling
    samples, timepoints = sim_new.build_samples_real_formethod(rna_arrays_sub_ancestors, rna_arrays_sub[1:])
    rna_arrays_method_res = MFL_method(models_method, samples, timepoints, weights_deconvoluted_sub_ancestors,
                                                                    weights_deconvoluted_sub[1:], n_iter*2)
    rna_arrays_sub_ancestors_res = [rna_arrays_method_res[2*i] for i in range(len(rna_arrays_sub_ancestors))]
    rna_arrays_sub_descendants_res = [rna_arrays_method_res[2*i+1] for i in range(len(rna_arrays_sub_ancestors))]

    if np.linalg.norm(rna_arrays_method_res[0]):

        samples, timepoints = sim_new.build_samples_real(rna_arrays_all)
        rna_arrays_all_res = MFL_basic(models_basic, samples, weights_deconvoluted_all, timepoints, n_iter)

        samples, timepoints = sim_new.build_samples_real(rna_arrays_sub)
        rna_arrays_sub_res = MFL_basic(models_basic, samples, weights_deconvoluted_sub, timepoints, n_iter)

    else:
        rna_arrays_all_res = rna_arrays_method_res
        rna_arrays_sub_res = rna_arrays_method_res

    return [[rna_arrays_all, weights_deconvoluted_all, rna_arrays_sub, weights_deconvoluted_sub,
           rna_arrays_sub_ancestors, weights_deconvoluted_sub_ancestors, rna_arrays_sub[1:]], [rna_arrays_all_res, weights_deconvoluted_all, rna_arrays_sub_res, weights_deconvoluted_sub,
           rna_arrays_sub_ancestors_res, weights_deconvoluted_sub_ancestors, rna_arrays_sub_descendants_res]]

def MFL_basic(models, samples, weights, timepoints, n_iter):
    """Function generating the trajectory inferred by the MFL algorithm published in [Zhang2022]"""

    number_cells_copy = number_cells[:len(np.unique(timepoints))]
    m = models.TrajLoss([torch.randn(number_cells_copy[i], samples.shape[1])*0.1 for i in range(0, len(number_cells_copy))],
                            [torch.tensor(weights[i], device = device) for i in range(0, len(number_cells_copy))],
                            torch.tensor(samples, device = device),
                            torch.tensor(timepoints, device = device),
                            dt = (T[-1] - T[0])/len(T), tau = diffusion_constant, sigma = None, M = number_cells_copy,
                            lamda_reg = torch.tensor(.05*np.ones(len(T)))/factor_MFL4, lamda_cst = 0, sigma_cst = float("Inf"),
                            branching_rate_fn = None,
                            sinkhorn_iters = 250, device = device, warm_start = True)

    models.optimize(m, n_iter = int(n_iter*1.5), eta_final = 0.1, tau_final = diffusion_constant,
                                   sigma_final = 0.5, N = np.mean(number_cells_copy), temp_init = 2*diffusion_constant, temp_ratio = 1.0,
                                   dim = samples.shape[1], tloss = m, print_interval = 50)

    models.optimize(m, n_iter = int(n_iter), eta_final = 0.1, tau_final = diffusion_constant,
                                   sigma_final = 0.5, N = np.mean(number_cells_copy), temp_init = diffusion_constant, temp_ratio = 1.0,
                                   dim = samples.shape[1], tloss = m, print_interval = 50)

    return [mt.detach().numpy() for mt in m.x]

def MFL_method(models, samples, timepoints, weights_anc, weights_desc, n_iter):

    """Function generating the trajectory inferred by the MFL-like algorithm detailed in the Appendix B of [Ventre2023]"""

    number_cells_copy = np.array([number_cells[0] for _ in range(0, len(np.unique(timepoints)))])
    weights = []
    for i in range(len(weights_anc)):
        weights.append(weights_anc[i])
        weights.append(weights_desc[i])
    print([samples[timepoints == i, :].shape for i in range(0, len(np.unique(timepoints)))])
    m = models.TrajLoss([torch.randn(number_cells_copy[i], samples.shape[1])*0.1 for i in range(0, len(number_cells_copy))],
                        [torch.tensor(weights[i], device = device) for i in range(0, len(weights))],
                        torch.tensor(samples, device = device),
                        torch.tensor(timepoints, device = device),
                        dt = (T[-1] - T[0])/len(T), tau = diffusion_constant, chain = chain, sigma = None,
                        M = number_cells_copy, lamda_unbal = None,
                        lamda_reg = torch.tensor(.05*np.ones(len(number_cells_copy)))/factor_MFL4, lamda_cst = 0, sigma_cst = float("Inf"),
                        branching_rate_fn = None,
                        sinkhorn_iters = 500, device = device, warm_start = False)

    models.optimize(m, n_iter = int(n_iter*1.5), eta_final = 0.1, tau_final = diffusion_constant,
                    sigma_final = 0.5, N = np.mean(number_cells_copy),
                    temp_init = diffusion_constant*2, temp_ratio = 1.0,
                    dim = samples.shape[1], tloss = m, print_interval = 50)

    obj, pbj_primal, x_res = models.optimize(m, n_iter = int(n_iter), eta_final = 0.1, tau_final = diffusion_constant,
                    sigma_final = 0.5, N = np.mean(number_cells_copy),
                    temp_init = diffusion_constant, temp_ratio = 1.0,
                    dim = samples.shape[1], tloss = m, print_interval = 50)

    return [mt.detach().numpy() for mt in x_res]

def build_fig(axes, res_list, weight_bool):

    if not weight_bool:
        leg = ["A", "B", "C", "D"]
    else:
        leg = ["E", "F", "G", "H"]

    for i in range(0, 3):
        axes[i].set_title("{}".format(leg[i]), weight="bold")
        if weight_bool: axes[i].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)
        if not i: axes[i].set_ylabel('Gene ' + str(dimensions_to_plot[1] + 1), fontsize=12)
        if i < 2: tmp, ttmp = sim_new.build_samples_real(res_list[0][2*i])
        else: tmp, ttmp = sim_new.build_samples_real(res_list[0][4] + [res_list[0][6][-1]])
        s = axes[i].scatter(tmp[:, dimensions_to_plot[0]],tmp[:, dimensions_to_plot[1]], c = ttmp, alpha=.5)
        axes[i].set_ylim(-2, 1)

    tmp_diff_ref = []
    tmp_diff_without = []
    tmp_diff_with = []
    for m, res in enumerate(res_list):
        tmp_ref, tmp_without, tmp_with = 0, 0, 0
        for t in range(0, len(T[:-1])):
            for n in range(0, n_sim):
                if not weight_bool:
                    sim_ref = simulate_groundtruth(M, res[0][t], flow_type, T[t], [T[t+1] - T[t]], weights=res[1][t])[0]
                    sim_without = simulate_groundtruth(M, res[2][t], flow_type, T[t], [T[t+1] - T[t]], weights=res[3][t])[0]
                    sim_with = simulate_groundtruth(M, res[4][t], flow_type, T[t], [T[t+1] - T[t]], weights=res[5][t])[0]
                    tmp_ref += RMS(sim_ref, res[0][t+1], nu_weight=res[1][t+1]) / n_sim
                    tmp_without += RMS(sim_without, res[2][t+1], nu_weight=res[3][t+1]) / n_sim
                    tmp_with += RMS(sim_with, res[6][t], nu_weight=res[3][t+1]) / n_sim
                else:
                    sim_ref = simulate_groundtruth(M, res[0][t], flow_type, T[t], [T[t+1] - T[t]])[0]
                    sim_without = simulate_groundtruth(M, res[2][t], flow_type, T[t], [T[t+1] - T[t]])[0]
                    sim_with = simulate_groundtruth(M, res[4][t], flow_type, T[t], [T[t+1] - T[t]])[0]
                    tmp_ref += RMS(sim_ref, res[0][t+1]) / n_sim
                    tmp_without += RMS(sim_without, res[2][t+1]) / n_sim
                    tmp_with += RMS(sim_with, res[6][t]) / n_sim

        print(tmp_ref, tmp_without, tmp_with)

        if not weight_bool:
            tmp_diff_ref.append(tmp_ref / len(T[:-1]))
            tmp_diff_without.append(tmp_without / len(T[:-1]))
            tmp_diff_with.append(tmp_with / len(T[:-1]))
        else:
            if np.linalg.norm(res[0][0]):
                tmp_diff_ref.append(tmp_ref / len(T[2:-1]))
                tmp_diff_without.append(tmp_without / len(T[2:-1]))
                tmp_diff_with.append(tmp_with / len(T[2:-1]))

    plt.xticks(fontsize=8)
    prop = {'color':'black'}
    test = axes[3].boxplot([tmp_diff_ref, tmp_diff_without, tmp_diff_with],
        patch_artist=True, medianprops=prop, widths=(.3, .3, .3))
    for patch in test['boxes']: patch.set(facecolor='white')
    axes[3].set_xlim(0.5,3.5)
    axes[3].set_ylim(0.25, 0.75)
    axes[3].set_ylabel('RMS', fontsize=12)
    axes[3].set_xticklabels(['no subsampling   ', '   no correction', 'corrected'], fontsize=12)
    axes[3].set_title("{}".format(leg[-1]), weight="bold")

    if weight_bool:
        axes[4].axis('off')
        cax = plt.axes([0.75, .11, 0.006, 0.33])
        cbar = fig.colorbar(s, ax=axes[4], cax=cax)
        cbar.ax.set_title('time', fontsize=12)
    else:
        axes[4].axis('off')

    return axes

# Build results
res = [[], []]
for m in range(n_mean):
    tmp = simulate_trees(N0, T, flow_type, p_vec)
    res[0].append(tmp[0])
    res[1].append(tmp[1])

lines = 2
fig, axes = plt.subplots(lines, 5, figsize=(25, 5*lines))
for i in range(0, lines):
    print(i)
    axes[i] = build_fig(axes[i], res[i], i)
plt.savefig("Figures/Figure6.pdf", dpi=150)
plt.close()
