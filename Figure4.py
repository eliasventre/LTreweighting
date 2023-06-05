import os
num_threads = "8"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import copy
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
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

flow_types = ['MFL_2']

# Setting simulation parameters
timescale = 1
tau, division_time_distribution = .15, "exponential"
device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)

# Choosing which of the three dimensions to show in later plots
dimensions_to_plot = [0, 1]

n_mean = 1
N_trees = 15
M = np.array([100, 200], dtype=int)
n_iter = 500

x_init = np.array([0, 0, 0])
T = np.linspace(.01, 1.1, 7)
non_constant_branching_rate = True
mean_division_time = .25*timescale


def Sinkhorn(mu, nu, K, n=1000):
    g = np.ones_like(nu)
    f = mu / np.dot(K, g)
    cnt = 0
    while cnt < n:
        g = nu / np.dot(K.T, f)
        f = mu / np.dot(K, g)
        res = np.outer(f, g) * K
        cnt += 1
    return res, f, g


def MFL_branching(N0, flow_type, M):

    sim_params = sim.SimulationParameters(division_time_std = .001*timescale,
                                          flow_type = flow_type,
                                          mutation_rate = 1/timescale,
                                          mean_division_time = mean_division_time,
                                          timestep = 0.005*timescale,
                                          initial_distribution_std = 0.1,
                                          diffusion_constant = tau,
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

    weights_deconvoluted = sim_new.build_weights_deconvoluted(true_trees_annotated, N0)

    m_with_treestructure = models.TrajLoss([torch.randn(M, samples_real.shape[1])*0.1 for i in range(0, len(T))],
                            [torch.tensor(weights_deconvoluted[i], device=device) for i in range(0, len(T))],
                            torch.tensor(samples_real, device = device),
                            torch.tensor(t_idx_real, device = device),
                            dt = (T[-1] - T[0])/len(T), tau = sim_params.diffusion_constant, sigma = None,
                                           M = M * np.ones_like(T, dtype=int),
                            lamda_reg = torch.tensor(.05*np.ones(len(T))), lamda_cst = 0, sigma_cst = float("Inf"),
                            branching_rate_fn = None,
                            sinkhorn_iters = 250, device = device, warm_start = True)

    models.optimize(m_with_treestructure, n_iter = n_iter, eta_final = 0.1, tau_final = sim_params.diffusion_constant,
                                   sigma_final = 0.5, N = M, temp_init = 1.0, temp_ratio = 1.0,
                                   dim = samples_real.shape[1], tloss = m_with_treestructure, print_interval = 50)

    return rna_arrays, weights_deconvoluted, m_with_treestructure.x


def build_vf_pr(tau, time_course_datasets, rna_arrays, weights_deconvoluted):
    vf, pr = [], []
    dim = np.size(rna_arrays[0], 1)
    for tmpt in range(0, len(time_course_datasets)-1):

        data1, data2 = time_course_datasets[tmpt], time_course_datasets[tmpt+1]
        pairwise_dists = cdist(data1, data2, 'sqeuclidean')
        K = np.exp(-pairwise_dists / (2 * tau * (T[tmpt+1] - T[tmpt])))
        mu, nu = np.ones(np.size(pairwise_dists, 0)) / np.size(pairwise_dists, 0), \
                 np.ones(np.size(pairwise_dists, 1)) / np.size(pairwise_dists, 1)
        for i in range(0, np.size(mu)):
            K[i, :] /= np.sum(K[i, :])
        P, f, g = Sinkhorn(mu, nu, K)

        vf_res = np.ones((np.size(pairwise_dists, 0), dim))
        for i in range(0, np.size(mu)):
            P[i, :] /= np.sum(P[i, :])
            for d in range(dim):
                vf_res[i, d] = np.sum(P[i, :] * (data2 - data1[i])[:, d])
        vf.append(vf_res / (T[tmpt+1] - T[tmpt]))

        M1 = cdist(rna_arrays[tmpt], data1, 'sqeuclidean')
        M2 = cdist(rna_arrays[tmpt+1], data2, 'sqeuclidean')

        pi1 = ot.emd(weights_deconvoluted[tmpt], mu, M1)
        pi2 = ot.emd(weights_deconvoluted[tmpt+1], nu, M2)
        tmp1 = 1/(N_trees*weights_deconvoluted[tmpt])
        tmp2 = 1/(N_trees*weights_deconvoluted[tmpt+1])
        for i in range(np.size(mu)):
            mu[i] = (np.sum((tmp1) * pi1[:, i]))
            mu[i] += np.exp2(np.sum(np.log2(tmp1) * pi1[:, i] / np.sum(pi1[:, i]))) * np.sum(pi1[:, i])
        for j in range(np.size(nu)):
            nu[j] = (np.sum((tmp2) * pi2[:, j]))
            nu[j] += np.exp2(np.sum(np.log2(tmp2) * pi2[:, j] / np.sum(pi2[:, j]))) * np.sum(pi2[:, j])

        res = np.log(np.sum(P * np.outer(1/mu, nu), 1)) / (T[tmpt+1] - T[tmpt])
        M_reg = cdist(data1, data1, 'sqeuclidean')
        Ker = np.exp(-M_reg/tau)
        Ker /= np.sum(Ker, 1)
        res = np.dot(Ker, res)
        pr.append(res)

    return vf, pr

def build_true_vf_pr(rna_arrays, flow_type):
    sim_params = sim.SimulationParameters(flow_type = flow_type, mean_division_time=mean_division_time)
    vf, pr = [], []
    dim = rna_arrays[0].shape[1]
    for tmpt in range(0, len(rna_arrays)-1):
        vf_res = np.ones_like(rna_arrays[tmpt])
        pr_res = np.ones(np.size(vf_res, 0))
        for x in range(np.size(vf_res, 0)):
            vf_res[x, :] = sim.vector_field(rna_arrays[tmpt][x, :], T[tmpt], sim_params)
            pr_res[x] = 1/sim.func_mean_division_time(rna_arrays[tmpt][x, :].reshape(1, dim), sim_params)[0]
        vf.append(vf_res)
        pr.append(pr_res)
    return vf, pr

def build_fig(i, axes, flow_type, rna_arrays_new, pr_tr, pr_ref_tr, vf_tr, vf_ref_tr):

    rms_vf_tree = list()
    rms_pr_tree = list()

    for tmpt in range(0, len(T)-1):

        rms_vf_tmp = []
        rms_pr_tmp = []
        for n in range(n_mean):
            rms_vf_tmp.append(RMS(vf_ref_tr[n][tmpt], vf_tr[n][tmpt]))
            rms_pr_tmp.append(RMS(pr_ref_tr[n][tmpt].reshape(len(pr_ref_tr[n][tmpt]), 1),
                                  pr_tr[n][tmpt].reshape(len(pr_tr[n][tmpt]), 1)))
        rms_vf_tree.append(np.mean(rms_vf_tmp))
        rms_pr_tree.append(np.mean(rms_pr_tmp))

    axes[0].set_title("A", weight="bold")
    axes[0].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1))
    if not i: axes[0].set_ylabel('Gene ' + str(dimensions_to_plot[1] + 1))
    samples_real_new, t_idx_real_new = sim_new.build_samples_real(rna_arrays_new[0][:-1])
    velocity_tree, t_idx_real = sim_new.build_samples_real(vf_tr[0])
    velocity_tree_true, t_idx_real_true = sim_new.build_samples_real(vf_ref_tr[0])
    axes[0].quiver(samples_real_new[:, dimensions_to_plot[0]],samples_real_new[:, dimensions_to_plot[1]],
                    velocity_tree[:, dimensions_to_plot[0]],velocity_tree[:, dimensions_to_plot[1]])
    axes[0].quiver(samples_real_new[:, dimensions_to_plot[0]],samples_real_new[:, dimensions_to_plot[1]],
                    velocity_tree_true[:, dimensions_to_plot[0]],velocity_tree_true[:, dimensions_to_plot[1]], color="red")
    axes[0].set_ylim(-1.3, .6)
    axes[0].set_xlim(-1.7, 1.7)

    N = 100
    y = np.linspace(-1.3, .6, N)
    x = np.linspace(-1.7, 1.7, N)
    z = np.outer(x, y)
    sim_params = sim.SimulationParameters(division_time_std = .001*timescale,
                                          flow_type = flow_type,
                                          mutation_rate = 1/timescale,
                                          mean_division_time = mean_division_time,
                                          timestep = 0.005*timescale,
                                          initial_distribution_std = 0.1,
                                          diffusion_constant = tau,
                                          division_time_distribution = division_time_distribution,
                                          non_constant_branching_rate = non_constant_branching_rate
                                         )
    for k, yy in enumerate(y):
        for l, xx in enumerate(x):
            z[k,l] = 1/sim.func_mean_division_time(np.array([xx, yy, 0]).reshape(1, 3), sim_params)[0]
    levels = ticker.MaxNLocator(nbins=100).tick_values(np.min(z), np.max(z))
    axes[1].contourf(x, y, z, levels=levels, alpha=.2)
    axes[1].set_title("B", weight="bold")

    axes[1].set_xlabel('Gene ' + str(dimensions_to_plot[0] + 1))
    birth_tree, t_idx_real = sim_new.build_samples_real(pr_tr[0])
    bmax = np.quantile(birth_tree, .9)
    bmin = min(0, np.quantile(birth_tree, .1))
    s_inf2 = axes[1].scatter(samples_real_new[:, dimensions_to_plot[0]],samples_real_new[:, dimensions_to_plot[1]], alpha = .7,
                    c = birth_tree * (birth_tree > bmin) * (birth_tree < bmax) + (bmin) * (birth_tree <= bmin) + bmax * (birth_tree >= bmax))

    if not i:
        axes[2].axis('off')
        cax = plt.axes([0.66, .11, 0.01, 0.75])
        cbar = fig.colorbar(s_inf2, ax=axes[2], cax=cax)
        cbar.ax.set_title('rate')

    return axes


res = []
for i, ft in enumerate(flow_types):
    res.append([])
    for _ in range(n_mean):
        res[i].append(MFL_branching(N_trees, ft, M[i]))

velocity_fields_trees = []
proliferation_rates_trees = []
velocity_fields_true_trees = []
proliferation_rates_true_trees = []
sample_trees_new = []
sample_trees_init = []

for f in range(0, len(flow_types)):
    pr_tr, pr_ref_tr, vf_tr, vf_ref_tr = [],[],[],[]
    sample_trees_tmp = []
    sample_real_tmp = []

    for n in range(n_mean):

        samples_real, mult_div, m_trees = res[f][n]

        time_course_datasets_tree = [m.detach().numpy() for m in m_trees]
        tmp_vel, tmp_prol = build_vf_pr(tau, time_course_datasets_tree, samples_real, mult_div)
        vf_tr.append(tmp_vel)
        pr_tr.append(tmp_prol)

        tmp_vel, tmp_prol = build_true_vf_pr(time_course_datasets_tree, flow_types[f])
        vf_ref_tr.append(tmp_vel)
        pr_ref_tr.append(tmp_prol)

        sample_trees_tmp.append(time_course_datasets_tree)
        sample_real_tmp.append(samples_real)

    sample_trees_new.append(sample_trees_tmp)
    sample_trees_init.append(sample_real_tmp)

    velocity_fields_trees.append(vf_tr)
    proliferation_rates_trees.append(pr_tr)
    velocity_fields_true_trees.append(vf_ref_tr)
    proliferation_rates_true_trees.append(pr_ref_tr)


fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i, flow_type in enumerate(flow_types):
    axes = build_fig(i, axes, flow_type, sample_trees_new[i], proliferation_rates_trees[i],
                     proliferation_rates_true_trees[i], velocity_fields_trees[i], velocity_fields_true_trees[i])

    plt.savefig("Figure4.pdf", dpi=150)
    plt.close()
    plt.show()
