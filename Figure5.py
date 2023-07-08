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

import sys; sys.path += ['../.']

from lineageot_files import simulation as sim
from lineageot_files import inference as sim_inf
from lineageot_files import new as sim_new
from lineageot_files.new import RMS, subsampled_tree

np.random.seed(2)

flow_type = 'MFL_4' # flow type used for the simulations
factor_MFL4 = 4

# Setting simulation parameters
timescale = 1
diffusion_constant, division_time_distribution = factor_MFL4*.2, "exponential"
mean_division_time = .4*timescale
mean_death_time = 1*timescale
x_init = np.array([0, 0.15, 0])
dim = 3

# Choosing which of the three dimensions to show in later plots
dimensions_to_plot = [0, 1]
T = np.linspace(.1, 1.5, 7) # time of simulation

n_mean = 7 # number of repetitions
p_vec = np.ones(len(T))
p_vec[-2:] = [.8, .5] # initial subsampling probability vector
n_ps = [1, 1, .5, .1, 0.01] # list of multiplicative factors for the subsampling probability
N0 = 10 # number of initial cells



def simulate_groundtruth(N0, T, flow_type):
    """Function simulating N0 cells at a sequence of timepoints T under a SDE without branching and drift = flow_type """
    non_constant_branching_rate = False
    mean_division_time = 1e9
    sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                          flow_type = flow_type,
                                          mutation_rate = 1/timescale,
                                          mean_division_time = mean_division_time,
                                          timestep = 0.01*timescale,
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

    return rna_arrays



def simulate_trees(N0, T, flow_type, proba_sub, true_trees=None, compute=1):
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
    if compute:
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
        true_trees = [[] for _ in range(0, len(T))]
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

    if not compute:
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

    else: rna_arrays_sub, weights_deconvoluted_sub = rna_arrays_all, weights_deconvoluted_all

    print(rna_arrays_all[-1].shape, np.mean([rna_arrays_sub[i].shape[0] for i in range(len(rna_arrays_sub))]))
    return [rna_arrays_all, weights_deconvoluted_all, rna_arrays_sub, weights_deconvoluted_sub, true_trees]


res = [[] for _ in range(0, n_mean)]
tree, rna = 0, 0
for m in range(n_mean):
    for n, p in enumerate(n_ps):
        print(m, n)
        if not n:
            tmp = simulate_trees(N0, T, flow_type, np.ones(len(T)))
            tree = tmp[-1]
        else:
            tmp = simulate_trees(N0, T, flow_type, p_vec * (np.linspace(max(np.sqrt(p), .5), np.sqrt(p), len(T)))**2,
                                 true_trees=tree, compute=0)
        res[m].append(tmp)

ground_truth = simulate_groundtruth(2500, T, flow_type)

lines = 1
col = 5
fig, axes = plt.subplots(lines, col, figsize=(5*col, 5*lines))

x_obs, t_obs = sim_new.build_samples_real(ground_truth)
axes[2].set_title("C", weight="bold")
axes[2].scatter(x_obs[:, dimensions_to_plot[0]],x_obs[:, dimensions_to_plot[1]], c = t_obs, alpha=.5)
axes[2].set_ylim(-2, 1)
axes[2].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)

rna = [sim_new.build_samples_real(res[-1][-2][0]),
       sim_new.build_samples_real(res[-1][-2][2])]

axes[0].set_title("A", weight="bold")
axes[0].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)
axes[0].set_ylabel('Gene ' + str(dimensions_to_plot[1] + 1), fontsize=12)
s = axes[0].scatter(rna[0][0][:, dimensions_to_plot[0]],rna[0][0][:, dimensions_to_plot[1]], c = rna[0][1], alpha=.5)
axes[0].set_ylim(-2, 1)

axes[1].set_title("B", weight="bold")
axes[1].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)
axes[1].scatter(rna[1][0][:, dimensions_to_plot[0]],rna[1][0][:, dimensions_to_plot[1]], c = rna[1][1], alpha=.5)
axes[1].set_ylim(-2, 1)

tmp_diff_ref = [[0 for m in range(0, n_mean)] for _ in n_ps]
tmp_diff_with = [[0 for m in range(0, n_mean)] for _ in n_ps]
for p in range(0, len(n_ps)):
    for m in range(0, n_mean):
        for t in range(0, len(T)):
            rna_nosub = res[m][p][0][t]
            rna_sub = res[m][p][2][t]
            tmp_diff_ref[p][m] += RMS(rna_nosub, x_obs[t_obs == t], mu_weight=res[m][p][1][t]) / len(T)
            tmp_diff_with[p][m] += RMS(rna_sub, x_obs[t_obs == t], mu_weight=res[m][p][3][t]) / len(T)
diff_noreweighting = 0
for m in range(0, n_mean):
    for t in range(0, len(T)):
        diff_noreweighting += RMS(res[m][-2][0][t], x_obs[t_obs == t])
diff_noreweighting /= n_mean*len(T)

plt.xticks(fontsize=8)
prop = {'color':'black'}
test = axes[3].boxplot([tmp_diff_with[p] for p in range(0, len(n_ps))],
    patch_artist=True, medianprops=prop, widths=([.2 for _ in range(0, len(n_ps))]))
for patch in test['boxes']: patch.set(facecolor='white')
axes[3].set_xlim(0.5, len(n_ps)+.5)
axes[3].set_ylabel('RMS', fontsize=12)
axes[3].set_xlabel('subsampling rate', fontsize=12)
list_tmp = ['{}'.format(p) for p in n_ps]
list_tmp[1] = '0.8'
axes[3].set_xticklabels(list_tmp)
axes[3].plot(np.arange(len(n_ps)+2), [diff_noreweighting for _ in range(len(n_ps)+2)], '--', c='grey')
axes[3].set_title("D", weight="bold")

axes[4].axis('off')
cax = plt.axes([0.75,.11, 0.006, 0.74])
cbar = fig.colorbar(s, ax=axes[2], cax=cax)
cbar.ax.set_title('time', fontsize=12)


plt.savefig("Figures/Figure5.pdf", dpi=150)
plt.close()
