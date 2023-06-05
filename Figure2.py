import os

import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import sys; sys.path += ['../']

import lineageot_files.simulation as sim
import lineageot_files.inference as sim_inf
import lineageot_files.new as sim_new

flow_types = ['MFL_2', 'MFL_4'] # flow type used for the simulations

np.random.seed(2)

# Setting simulation parameters
timescale = 1
diffusion_constant, division_time_distribution = [.2, .8], "exponential"

# Choosing which of the three dimensions to show in later plots
dimensions_to_plot = [0, 1]

N0 = 2500 # number of trees/cells to simulate
N_list = [1, 2, 4, 7, 10, 15, 20, 25]
n_mean = np.ones(len(N_list))
T = [1.5] # time of simulation

def RMS(mu, nu, mu_weight, nu_weight, bool):
    """Function measuring the energy distance between two empirical distributions """
    res = 0
    for i in range(0, mu.shape[1]):
        res += sp.stats.energy_distance(mu[:, i], nu[:, i], u_weights=mu_weight, v_weights=nu_weight)**2
    return np.sqrt(res)


def simulate_SDE(N0, T, flow_type, i):
    """Function simulating N0 cells at a sequence of timepoints T under a SDE without branching and drift = flow_type """
    non_constant_branching_rate = False # False if the branching rate is constant
    non_constant_death_rate = False # False if the branching rate is constant
    mean_division_time = 1e9 # no birth
    mean_death_time = 1e9 # no death
    sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                        flow_type = flow_type,
                                        mutation_rate = 1/timescale,
                                        mean_division_time = mean_division_time,
                                        mean_death_time = mean_death_time,
                                        timestep = 0.02*timescale,
                                        initial_distribution_std = 0.1,
                                        diffusion_constant = diffusion_constant[i],
                                        division_time_distribution = division_time_distribution,
                                        non_constant_branching_rate = non_constant_branching_rate,
                                        non_constant_death_rate = non_constant_death_rate,
                                        )
    # Simulating a set of samples
    sample = []
    initial_cell = []
    for cnt, t in enumerate(T):
        tmp = []
        tmp_initial_cell = []
        for n in range(0, N0):
            tmp_initial_cell.append(sim.Cell(np.random.normal(np.array([0, 0.15, 0]), sim_params.initial_distribution_std),
                                    np.zeros(sim_params.barcode_length)))
            tmp.append(sim.sample_descendants(tmp_initial_cell[n], t, t, sim_params))
        sample.append(tmp)
        initial_cell.append(tmp_initial_cell)

    # Extracting trees and barcode matrices
    true_trees = [[] for _ in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            true_trees[t].append(sim_inf.list_tree_to_digraph(sample[t][n]))
    # Computing array of cells
    data_arrays = [[] for _ in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            data_arrays[t].append(sim_inf.extract_data_arrays(true_trees[t][n]))
    rna_arrays = []
    for t in range(0, len(T)):
        tmp_array = data_arrays[t][0][0]
        for n in range(1, N0):
            tmp_array = np.concatenate((tmp_array, data_arrays[t][n][0]), axis=0)
        rna_arrays.append(tmp_array)

    return rna_arrays


def simulate_trees(N0, T, flow_type, i):
    """Function simulating N0 trees at a sequence of timepoints T under a branchgin SDE, drift = flow_type
     and branching rates associated to the same flow_type """
    non_constant_branching_rate = True # False if the branching rate is constant
    non_constant_death_rate = False # False if the branching rate is constant
    mean_division_time = .4 # 1 / birth rate
    mean_death_time = 1e9 # no death
    sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                        flow_type = flow_type,
                                        mutation_rate = 1/timescale,
                                        mean_division_time = mean_division_time,
                                        mean_death_time = mean_death_time,
                                        timestep = 0.02*timescale,
                                        initial_distribution_std = 0.1,
                                        diffusion_constant = diffusion_constant[i],
                                        division_time_distribution = division_time_distribution,
                                        non_constant_branching_rate = non_constant_branching_rate,
                                        non_constant_death_rate = non_constant_death_rate,
                                        )
    # Simulating a set of samples
    sample = []
    initial_cell = []
    for cnt, t in enumerate(T):
        tmp = []
        tmp_initial_cell = []
        for n in range(0, N0):
            tmp_initial_cell.append(sim.Cell(np.random.normal(np.array([0, 0.15, 0]), sim_params.initial_distribution_std),
                                    np.zeros(sim_params.barcode_length)))
            tmp.append(sim.sample_descendants(tmp_initial_cell[n], t, t, sim_params))
        sample.append(tmp)
        initial_cell.append(tmp_initial_cell)

    # Extracting trees and barcode matrices
    true_trees = [[] for t in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            true_trees[t].append(sim_inf.list_tree_to_digraph(sample[t][n]))
    # Computing array of cells
    data_arrays = [[] for t in range(0, len(T))]
    for t in range(0, len(T)):
        for n in range(0, N0):
            data_arrays[t].append(sim_inf.extract_data_arrays(true_trees[t][n]))
    rna_arrays = []
    for t in range(0, len(T)):
        tmp_array = data_arrays[t][0][0]
        for n in range(1, N0):
            tmp_array = np.concatenate((tmp_array, data_arrays[t][n][0]), axis=0)
        rna_arrays.append(tmp_array)

    # Extract weights

    true_trees_annotated = [[copy.deepcopy(true_trees[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    for n in range(0, N0):
        for t in range(0, len(T)):
            sim_inf.add_node_times_from_division_times(true_trees_annotated[t][n])
            for tt in range(0, t):
                sim_inf.add_nodes_at_time(true_trees_annotated[t][n], T[tt])

    weights_deconvoluted = sim_new.build_weights_deconvoluted(true_trees_annotated, N0)

    return rna_arrays, weights_deconvoluted, true_trees_annotated


def build_fig(i, axes, flow_type):

    res_rms = list()
    dist_rms = list()
    bad_rms = []
    N_leaves = list()
    rna_arrays_ref = simulate_SDE(N0, T, flow_type, i)
    print(rna_arrays_ref[-1].shape[0])
    weights = np.ones(rna_arrays_ref[-1].shape[0])/rna_arrays_ref[-1].shape[0]
    
    for n in N_list:
        res_rms_tmp = []
        dist_rms_tmp = []
        bad_rms_tmp = []
        N_leaves_tmp = []
        for k in range(0, 5 + int(100/n)):
            rna_arrays_trees, weights_deconvoluted, true_trees = simulate_trees(n, T, flow_type, i)
            rna_arrays = simulate_SDE(1 + int(np.sum(rna_arrays_trees[-1][:, 0] <= 0)*(6*i + 2*(1-i))), T, flow_type, i) # We divide by the number of wells
            res_rms_tmp.append(RMS(rna_arrays_ref[-1], rna_arrays_trees[-1], weights, weights_deconvoluted[-1], i))
            dist_rms_tmp.append(RMS(rna_arrays_ref[-1], rna_arrays[-1], weights,
                            np.ones(rna_arrays[-1].shape[0])/rna_arrays[-1].shape[0], i))
            bad_rms_tmp.append(RMS(rna_arrays_ref[-1], rna_arrays_trees[-1], weights,
                            np.ones(rna_arrays_trees[-1].shape[0])/rna_arrays_trees[-1].shape[0], i))
            N_leaves_tmp.append(np.size(weights_deconvoluted[-1]))
        res_rms.append(np.median(res_rms_tmp))
        dist_rms.append(np.median(dist_rms_tmp))
        bad_rms.append(np.median(bad_rms_tmp))
        N_leaves.append(np.median(N_leaves_tmp))
        print(n, N_leaves)
    
    p1, = axes[2].plot(N_list, res_rms, c = "green")
    p1.set_label('BrSDE reweighted - SDE ref')
    p2, = axes[2].plot(N_list, dist_rms, c = 'blue')
    p2.set_label('SDE - SDE ref')
    p0, = axes[2].plot(N_list, bad_rms, c = 'red')
    p0.set_label('BrSDE - SDE ref')
    if i: axes[2].legend()
    axes[2].set_xlabel('Number of trees')
    axes[2].set_ylabel('RMS')

    samples_real, t_idx_real = sim_new.build_samples_real(rna_arrays_ref)
    axes[0].scatter(samples_real[:, dimensions_to_plot[0]], samples_real[:, dimensions_to_plot[1]],
                    c= "black", alpha = 0.25)
    if not i: axes[0].set_title("Samples SDE ref")
    axes[0].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1))
    axes[0].set_ylabel('Gene ' + str(dimensions_to_plot[1] + 1))
    # axes[0].set_xlim(np.min(samples_real[:, dimensions_to_plot[0]])-.1, np.max(samples_real[:, dimensions_to_plot[0]])+.1)
    axes[0].set_ylim(np.min(samples_real[:, dimensions_to_plot[1]])-.1, np.max(samples_real[:, dimensions_to_plot[1]])+.1)

    samples_real_tree, t_idx_real = sim_new.build_samples_real(rna_arrays_trees)
    axes[1].scatter(samples_real_tree[:, dimensions_to_plot[0]], samples_real_tree[:, dimensions_to_plot[1]],
                    c= "black", alpha = 0.25)
    if not i: axes[1].set_title("Samples BrDSDE")
    axes[1].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1))
    # axes[1].set_xlim(np.min(samples_real[:, dimensions_to_plot[0]])-.1, np.max(samples_real[:, dimensions_to_plot[0]])+.1)
    axes[1].set_ylim(np.min(samples_real[:, dimensions_to_plot[1]])-.1, np.max(samples_real[:, dimensions_to_plot[1]])+.1)
    return axes

fig, axes = plt.subplots(2, 3, figsize=(15,10))
for i, flow_type in enumerate(flow_types):
    axes[i, :] = build_fig(i, axes[i, :], flow_type)
plt.savefig("Figure2.pdf", dpi=150)
