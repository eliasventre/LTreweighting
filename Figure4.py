import os
num_threads = "8"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import copy
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import colors
from scipy.spatial.distance import cdist
import numpy as np
import ot
import torch

import sys; sys.path += ['../']

import lineageot_files.simulation as sim
import lineageot_files.inference as sim_inf
import lineageot_files.new as sim_new
from lineageot_files.new import RMS

import models_mfl as models

np.random.seed(2)

alpha = .5
independent = 1

flow_type = 'MFL_2_F4'

# Setting simulation parameters
timescale = 1
tau, division_time_distribution = .15, "exponential"
device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)

# Choosing which of the three dimensions to show in later plots
dimensions_to_plot = [0, 1]

N_trees = 15
M = 100
n_iter = 500

x_init = np.array([0, 0, 0])
T = np.linspace(.01, 1.1, 7)
non_constant_branching_rate = True
mean_division_time = .25*timescale

sim_params = sim.SimulationParameters(division_time_std = .01*timescale,
                                          flow_type = flow_type,
                                          mutation_rate = 1/timescale,
                                          mean_division_time = mean_division_time,
                                          timestep = 0.005*timescale,
                                          initial_distribution_std = 0.1,
                                          diffusion_constant = tau,
                                          division_time_distribution = division_time_distribution,
                                          non_constant_branching_rate = non_constant_branching_rate
                                         )


def Sinkhorn(mu, nu, K, n=1000):
    g = np.ones_like(nu)
    f = mu / np.dot(K, g)
    cnt = 0
    while cnt < n:
        g = nu / np.dot(K.T, f)
        f = mu / np.dot(K, g)
        res = np.outer(f, g) * K
        cnt += 1
    return res


def MFL_branching(N0, flow_type, M):

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

    # Creating a copy of the true tree
    true_trees_annotated = [[copy.deepcopy(true_trees[t][n]) for n in range(0, N0)] for t in range(0, len(T))]
    for n in range(0, N0):
        for t in range(0, len(T)):
            sim_inf.add_node_times_from_division_times(true_trees_annotated[t][n])
            for tt in range(0, t):
                sim_inf.add_nodes_at_time(true_trees_annotated[t][n], T[tt])

    weights_deconvoluted = sim_new.build_weights_deconvoluted(true_trees_annotated, N0)

    m_with_treestructure = models.TrajLoss([torch.randn(M, samples_real.shape[1])*0.1 for i in range(0, len(T))],
                            [torch.tensor(weights_deconvoluted[i], device=device) for i in range(0, len(T))],
                            torch.tensor(samples_real, device = device),
                            torch.tensor(t_idx_real, device = device),
                            dt = (T[-1] - T[0])/len(T), tau = sim_params.diffusion_constant, sigma = None,
                                           M = M * np.ones_like(T, dtype=int),
                            lamda_reg = torch.tensor(.05*np.ones(len(T))), lamda_cst = 0, sigma_cst = float("Inf"),
                            branching_rate_fn = None,
                            sinkhorn_iters = 500, device = device, warm_start = True)

    models.optimize(m_with_treestructure, n_iter = n_iter, eta_final = 0.1, tau_final = sim_params.diffusion_constant,
                                   sigma_final = 0.5, N = M, temp_init = 1.0, temp_ratio = 1.0,
                                   dim = samples_real.shape[1], tloss = m_with_treestructure, print_interval = 50)

    return rna_arrays, weights_deconvoluted, m_with_treestructure


def build_vf_pr(tau, time_course_datasets, rna_arrays, weights_deconvoluted):
    vf, pr = [], []
    dim = np.size(rna_arrays[0], 1)
    for tmpt in range(0, len(time_course_datasets)-1):

        data1, data2 = time_course_datasets[tmpt], time_course_datasets[tmpt+1]
        # plt.scatter(data1[:, 0], data1[:, 1])
        # plt.scatter(data2[:, 0], data2[:, 1], color='red')
        # plt.show()
        pairwise_dists = ot.dist(data1, data2)
        K = np.exp(-pairwise_dists / (4 * tau * (T[tmpt+1] - T[tmpt])))
        mu, nu = np.ones_like(K[:, 0])/np.size(K[:, 0]), np.ones_like(K[0, :])/np.size(K[0, :])
        P = Sinkhorn(mu, nu, K)
        P /= np.sum(P, 1)

        vf_res = (np.dot(P, data2) - data1) / (T[tmpt+1] - T[tmpt])

        M1 = cdist(rna_arrays[tmpt], data1, 'sqeuclidean')
        M2 = cdist(rna_arrays[tmpt+1], data2, 'sqeuclidean')

        pi1 = ot.sinkhorn(weights_deconvoluted[tmpt], mu, M1, reg=1/M)
        pi1 /= np.sum(pi1, 0)
        pi2 = ot.sinkhorn(weights_deconvoluted[tmpt+1], nu, M2, reg=1/M)
        pi2 /= np.sum(pi2, 0)
        tmp1 = 1/(N_trees*weights_deconvoluted[tmpt])
        tmp2 = 1/(N_trees*weights_deconvoluted[tmpt+1])
        p = 1/2 # balance between the average of the 2^m and 2^(average of m)
        for i in range(np.size(mu)):
            mu[i] = np.sum(tmp1 * pi1[:, i])*p
            mu[i] += np.exp2(np.sum(np.log2(tmp1) * pi1[:, i]))*(1-p)
        for j in range(np.size(nu)):
            nu[j] = np.sum(tmp2 * pi2[:, j])*p
            nu[j] += np.exp2(np.sum(np.log2(tmp2) * pi2[:, j]))*(1-p)

        pr_res = np.log(np.dot(P, nu)/mu) / (T[tmpt+1] - T[tmpt])

        #Regularization of the plot
        M_reg = ot.dist(data1, data1)
        Ker = np.exp(-M_reg/tau)
        Ker /= np.sum(Ker, 1)
        pr_res = np.dot(Ker, pr_res)
        vf_res = np.dot(Ker, vf_res)
        vf.append(vf_res)
        pr.append(pr_res)

    return vf, pr

def build_true_vf(rna_arrays, sim_params):
    vf, pr = [], []
    for tmpt in range(0, len(rna_arrays)-1):
        vf_res = np.ones_like(rna_arrays[tmpt])
        for x in range(np.size(vf_res, 0)):
            vf_res[x, :] = sim.vector_field(rna_arrays[tmpt][x, :], T[tmpt], sim_params)
        vf.append(vf_res)
    return vf

def build_fig(axes, rna_arrays, pr_tr, vf_tr):


    samples_real, t_idx_real_new = sim_new.build_samples_real(rna_arrays[:-1])
    velocity_tree, t_idx_real = sim_new.build_samples_real(vf_tr)
    vf_ref_tr = build_true_vf(time_course_datasets_tree, sim_params)
    velocity_tree_true, t_idx_real_true = sim_new.build_samples_real(vf_ref_tr)
    birth_tree, t_idx_real = sim_new.build_samples_real(pr_tr)

    axes[0, 0].set_title("A", weight="bold")
    axes[0, 0].set_xlabel('true velocity', fontsize=12)
    axes[0, 0].set_ylabel('inferred velocity', fontsize=12)
    axes[0, 0].scatter(velocity_tree_true[:, dimensions_to_plot[0]], velocity_tree[:, dimensions_to_plot[0]], marker='x', alpha=.5)
    min_tmp = -3
    max_tmp = 3
    axes[0, 0].plot([min_tmp, max_tmp], [min_tmp,max_tmp], c='grey', linestyle='dashed', alpha=.5)
    axes[0, 0].set_xlim(min_tmp, max_tmp)
    axes[0, 0].set_ylim(min_tmp, max_tmp)
    
    axes[0, 1].set_title("B", weight="bold")
    axes[0, 1].set_xlabel('true velocity', fontsize=12)
    axes[0, 1].scatter(velocity_tree_true[:, dimensions_to_plot[1]], velocity_tree[:, dimensions_to_plot[1]], marker='x', alpha=.5)
    min_tmp = -1.5
    max_tmp = 0.4
    axes[0, 1].set_xlim(min_tmp, max_tmp)
    axes[0, 1].set_ylim(min_tmp, max_tmp)
    axes[0, 1].plot([min_tmp,max_tmp], [min_tmp,max_tmp], c='grey', linestyle='dashed', alpha=.5)

    axes[1, 0].set_title("C", weight="bold")
    axes[1, 0].set_xlabel('gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)
    axes[1, 0].set_ylabel('gene ' + str(dimensions_to_plot[1] + 1), fontsize=12)
    axes[1, 0].quiver(samples_real[:, dimensions_to_plot[0]],samples_real[:, dimensions_to_plot[1]],
                    velocity_tree[:, dimensions_to_plot[0]], velocity_tree[:, dimensions_to_plot[1]], alpha =1)
    axes[1, 0].quiver(samples_real[:, dimensions_to_plot[0]],samples_real[:, dimensions_to_plot[1]],
                    velocity_tree_true[:, dimensions_to_plot[0]],velocity_tree_true[:, dimensions_to_plot[1]], color="red", alpha=.5)
    axes[1, 0].set_xlim(-2, 2)
    axes[1, 0].set_ylim(-1, .4)

    N = 100
    y = np.linspace(-1, .4, N)
    x = np.linspace(-2, 2, N)
    z = np.outer(x, y)
    for k, yy in enumerate(y):
        for l, xx in enumerate(x):
            z[k,l] = 1/sim.func_mean_division_time(np.array([xx, yy, 0]).reshape(1, 3), sim_params)[0]

    axes[1, 1].set_xlabel('gene ' + str(dimensions_to_plot[0] + 1), fontsize=12)
    bmax = np.quantile(birth_tree, .9)
    bmin = np.quantile(birth_tree, .1)

    levels = ticker.MaxNLocator(nbins=20).tick_values(np.min(z), np.max(z))
    s_inf = axes[1, 1].contourf(x, y, z, levels=levels, alpha=1)
    s_inf2 = axes[1, 1].scatter(samples_real[:, dimensions_to_plot[0]],samples_real[:, dimensions_to_plot[1]], alpha=1,
                    marker='o', edgecolors='black',
                    c = birth_tree * (birth_tree > bmin) * (birth_tree < bmax) + (bmin) * (birth_tree <= bmin) + bmax * (birth_tree >= bmax))
    axes[1, 1].set_title("D", weight="bold")

    axes[0, 2].axis('off')
    axes[1, 2].axis('off')
    cax = plt.axes([0.65, .11, 0.01, 0.34])
    cbar = fig.colorbar(s_inf, ax=axes[1, 2], cax=cax)
    cbar.ax.set_title('rate', fontsize=12)

    return axes


samples_real, weights_deconvoluted, m_trees = MFL_branching(N_trees, flow_type, M)
time_course_datasets_tree = [m.detach().numpy() for m in m_trees.x]
vf_tr, pr_tr = build_vf_pr(tau, time_course_datasets_tree, samples_real, weights_deconvoluted)


fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = build_fig(axes, time_course_datasets_tree, pr_tr, vf_tr)
plt.savefig("Figures/Figure4.pdf", dpi=150)
plt.close()
plt.show()
