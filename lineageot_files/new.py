from .inference import get_leaves, add_inverse_times_to_edges, remove_node_and_descendants
from .evaluation import expand_coupling
from .simulation import evolve_cell
import numpy as np
import scipy as sp
from scipy.linalg import expm
import ot
import matplotlib.pyplot as plt
import networkx as nx
import copy


def get_weight(tree, time, leaf = None):

    if leaf == None:
        leaves = [l for l in get_leaves(tree) if not l == 'root']
        weight = np.array([get_weight(tree, time, l) for l in leaves])
        return weight / np.sum(weight)
    else:
        weight = 1
        current_node = next(tree.predecessors(leaf))
        while tree.nodes[current_node]['time'] > time:
            weight *= 1 / len(list(nx.descendants_at_distance(tree, current_node, 1)))
            try: current_node = next(tree.predecessors(current_node))
            except: break
        return weight
    return


def cluster_leaves(tree, time, leaf = None):

    if leaf == None:
        leaves = [l for l in get_leaves(tree) if not l == 'root']
        return np.array([cluster_leaves(tree, time, l) for l in leaves])
    else:
        current_node = leaf
        while tree.nodes[current_node]['time'] > time:
            current_node = next(tree.predecessors(current_node))
        test = isinstance(current_node, tuple)
        if test:
            return current_node[1]
        return current_node
    return


def build_distances(trees, tmin = 0):
    res_total = []
    n_size = []
    for tree in trees:
        leaves = [l for l in get_leaves(tree) if not l == 'root']
        res = np.zeros((len(leaves), len(leaves)))
        for cnt, i in enumerate(leaves):
            current_node_i = next(tree.predecessors(i))
            list_node_from_i = [current_node_i]
            while 1:
                try:
                    current_node_i = next(tree.predecessors(current_node_i))
                    if type(current_node_i) == int:
                        list_node_from_i.append(current_node_i)
                except: break
            for j in leaves[cnt+1:]:
                current_node_j = next(tree.predecessors(j))
                while current_node_j not in list_node_from_i:
                    try: current_node_j = next(tree.predecessors(current_node_j))
                    except: break
                res[i, j] = max(0, tree.nodes[j]['time'] - tree.nodes[current_node_j]['time'] - tmin)
                res[j, i] = res[i, j]
        res_total.append(res)
        n_size.append(np.size(res, 0))
    distance_matrix = np.zeros((np.sum(n_size), np.sum(n_size)))
    n = 0
    for i in range(0, len(res_total)):
        n_new= n + n_size[i]
        distance_matrix[n:n_new, :][:, n:n_new] = res_total[i]
        n, m = n_new, n_new
    return distance_matrix


def diffuse_feature(dist_matrix, features, time_difference = 1):
    lapl = np.exp(-(dist_matrix / time_difference)**2)
    lapl /= np.sum(lapl, 1)
    lapl = .5*(lapl + lapl.T)
    lapl -= np.diag(np.diag(lapl))
    lapl -= np.sum(lapl, axis=1)
    eps = 1 # time_difference / 100
    res = expm(lapl * eps) @ features
    print(np.sum(np.abs(features - res)))
    return res


def eval_plot_metrics(axt, couplings, legends,
                      cost_func, Time, epsilons, scale):
    T, eps = len(Time)-1, len(epsilons)
    for t in range(0, T):
        y = [[] for _ in couplings]
        for e in range(eps):
            for cnt, coupling in enumerate(couplings):
                y[cnt].append(cost_func(coupling[t][e], t))
        # axt.set_title('Times coupling = {}-{}'.format(int(Time[t] * 100)/100, int(Time[t+1] * 100)/100))
        for cnt, coupling in enumerate(couplings):
            axt.plot(epsilons, np.array(y[cnt])/scale[t], label = legends[cnt])
        if not t:
            axt.set_ylabel("Normalized error")
            axt.legend()
        axt.set_xlabel("Entropy parameter")
        axt.set_xscale("log")
        axt.set_ylim(0, None)
    return axt


def build_samples_real(rna_arrays):
    T = np.arange(len(rna_arrays))
    res_samples = rna_arrays[0]
    res_times = np.zeros(rna_arrays[0].shape[0])
    for t in T[1:]:
        res_samples = np.concatenate((res_samples, rna_arrays[t]), axis=0)
        res_times = np.concatenate((res_times, t * np.ones(rna_arrays[t].shape[0])), axis=0)
    return res_samples, res_times


def build_samples_real_formethod(rna_arrays1, rna_arrays2):
    T = np.arange(len(rna_arrays1))

    res_samples = rna_arrays1[0]
    res_times = np.zeros(rna_arrays1[0].shape[0])

    for t in T[1:]:
        res_samples = np.concatenate((res_samples, rna_arrays2[t-1]), axis=0)
        res_times = np.concatenate((res_times, (2*t-1) * np.ones(rna_arrays2[t-1].shape[0])), axis=0)
        res_samples = np.concatenate((res_samples, rna_arrays1[t]), axis=0)
        res_times = np.concatenate((res_times, 2*t * np.ones(rna_arrays1[t].shape[0])), axis=0)

    res_samples = np.concatenate((res_samples, rna_arrays2[T[-1]]), axis=0)
    res_times = np.concatenate((res_times, (2*T[-1]+1) * np.ones(rna_arrays2[T[-1]].shape[0])), axis=0)

    return res_samples, res_times


def build_samples_deconvoluted(trees, times, rna_arrays, num_cells, N0):
    T = np.arange(len(rna_arrays))
    weight = []
    for t in T:
        weight.append(get_weight(trees[t][0], 0))
        for n in range(1, N0):
            weight[t] = np.concatenate((weight[t], get_weight(trees[t][n], 0)), axis=0)
    res_samples = rna_arrays[0][0:1, :]
    weight_min_tmp = np.min(weight[0])
    for c in range(0, num_cells[0]):
        for _ in range(int(weight[0][c]/weight_min_tmp)):
            res_samples = np.concatenate((res_samples, rna_arrays[0][c:c+1, :]), axis = 0)
    res_times = np.zeros(res_samples.shape[0])
    for t in T[1:]:
        cnt = 0
        print(t, np.var(weight[t]))
        weight_min_tmp = np.min(weight[t])
        for c in range(0, num_cells[t]):
            for _ in range(int(weight[t][c]/weight_min_tmp)):
                res_samples = np.concatenate((res_samples, rna_arrays[t][c:c+1, :]), axis = 0)
                cnt += 1
        res_times = np.concatenate((res_times, t * np.ones(cnt)), axis=0)
    return res_samples, res_times


def get_ancestors(tree, time):
    # get all leaf ancestors
    leaves = [l for l in get_leaves(tree) if not l == 'root']
    data = []
    list_ancestors = []
    for leaf in leaves:
        current_node = leaf
        while tree.nodes[current_node]['time'] > time:
            current_node = next(tree.predecessors(current_node))
        if tree.nodes[current_node]['time'] < time:
            error = "Tree has no ancestor of cell " + str(leaf) + " at time " + str(time)
            raise ValueError(error)
        if current_node not in list_ancestors:
            list_ancestors.append(current_node)
            data.append((tree.nodes[current_node]['x mean'], tree.nodes[current_node]['x variance']))
    return np.array([d[0] for d in data]), np.array([d[1] for d in data])

def get_weights_ancestors(tree, time):
    # get all leaf ancestors
    leaves = [l for l in get_leaves(tree) if not l == 'root']
    data = []
    list_ancestors = []
    for leaf in leaves:
        current_node = leaf
        while tree.nodes[current_node]['time'] > time:
            current_node = next(tree.predecessors(current_node))
        if tree.nodes[current_node]['time'] < time:
            error = "Tree has no ancestor of cell " + str(leaf) + " at time " + str(time)
            raise ValueError(error)

        if current_node not in list_ancestors:
            list_ancestors.append(current_node)
            weight = 1
            current_node = next(tree.predecessors(current_node))
            while tree.nodes[current_node]['time'] > 0:
                weight *= 1 / len(list(nx.descendants_at_distance(tree, current_node, 1)))
                try: current_node = next(tree.predecessors(current_node))
                except: break
            data.append(weight)
    return np.array(data)

def build_weights_deconvoluted(trees, N0):
    T = np.arange(len(trees))
    weight = []
    for t in T:
        weight.append(get_weight(trees[t][0], 0) / N0)
        for n in range(1, N0):
            weight[t] = np.concatenate((weight[t], get_weight(trees[t][n], 0) / N0), axis=0)

    return weight

def build_weights_deconvoluted_subsampled(trees, N0, empty):
    T = np.arange(len(trees))
    weight = []
    for t in T:
        weight.append(get_weight(trees[t][0], 0) / N0)
        for n in range(1, N0):
            if not empty[t][n]:
                weight[t] = np.concatenate((weight[t], get_weight(trees[t][n], 0) / N0), axis=0)

    return weight

def build_weights_deconvoluted_ancestors(trees, T, N0, empty):

    weight = []
    for t in range(1, len(T)):
        weight.append(get_weights_ancestors(trees[t][0], T[t-1]) / N0)
        for n in range(1, N0):
            if not empty[t][n]:
                weight[t-1] = np.concatenate((weight[t-1],
                                              get_weights_ancestors(trees[t][n], T[t-1]) / N0), axis=0)

    return weight


def build_weights(tree, time):

    cluster_leave = cluster_leaves(tree, time)
    weights = np.ones(len(cluster_leave))
    tmp = np.unique(cluster_leave, return_counts = True)
    for i in range(0, len(tmp[0])):
        weights[np.argwhere(cluster_leave == tmp[0][i])] = 1 / (tmp[1][i] * len(tmp[0]))
    return weights



### For filling trees

def bases_for_conditional_means_and_variances(tree, observed_nodes, time):

    node_list = [n for n in tree.nodes]
    add_inverse_times_to_edges(tree)
    l = nx.laplacian_matrix(nx.Graph(tree), nodelist = node_list, weight = 'inverse time')

    unobserved_nodes = []
    unobserved_node_indices = []
    observed_node_indices = []
    for c, n in enumerate(node_list):
        if n in observed_nodes and type(n) == int:
            observed_node_indices.append(c)
        else:
            unobserved_nodes.append(n)
            unobserved_node_indices.append(c)
    if len(observed_nodes) == 0:
        return

    conditioned_precision = np.array(l[np.ix_(unobserved_node_indices, unobserved_node_indices)].todense())
    conditioned_covariance = np.linalg.inv(conditioned_precision)

    mat_means = -1*(conditioned_covariance@l[np.ix_(unobserved_node_indices, observed_node_indices)])
    mat_covariance = np.zeros(len(observed_nodes))
    mat_rescale = np.zeros((len(observed_nodes), len(unobserved_nodes)))

    for cnt, l in enumerate(observed_nodes):
        if not l == 'root':
            current_node = l
            while tree.nodes[current_node]['time'] > time:
                current_node = next(tree.predecessors(current_node))
            if tree.nodes[current_node]['time'] < time:
                error = "Tree has no ancestor of cell " + str(l) + " at time " + str(time)
                raise ValueError(error)
            a = 0
            for i, node in enumerate(unobserved_nodes):
                if current_node == node:
                    a = i
                    break
            mat_rescale[cnt, a] = 1
            mat_covariance[cnt] = conditioned_covariance[a, a]

    return mat_means, mat_rescale, mat_covariance

def RMS(mu, nu, mu_weight=None, nu_weight=None):
    """Function measuring the energy distance between two empirical distributions """
    res = 0
    for i in range(0, mu.shape[1]):
        res += sp.stats.energy_distance(mu[:, i], nu[:, i], u_weights=mu_weight, v_weights=nu_weight)**2
    return np.sqrt(res)


def getleaves_of_node_digraph(G,node):
    # Finds leaves which are descendants of node in digraph G
    leaves=[]
    if G.out_degree(node)<1:
        leaves.append(node)
    else:
        for child in G.successors(node):
            leaves= leaves + getleaves_of_node_digraph(G,child)
    return leaves

def find_child(G,node,leaf):
    #finds immediate child of node which has leaf as descendant
    for child in G.successors(node):
        if leaf in getleaves_of_node_digraph(G,child):
            return child

def sample_leaves(tree, p):
    empty = 0
    sampled_cells=[]
    leaves=getleaves_of_node_digraph(tree,'root')
    ber = np.random.binomial(1, p, len(leaves))
    for i, leaf in enumerate(leaves):
        if ber[i] == 1:
            sampled_cells.append(leaf)
    if not len(sampled_cells):
        sampled_cells.append(leaves[0])
        empty = 0 # if empty = 1, this is not going to be taken into account later
    return sampled_cells, empty



def subsampled_tree(tree, p):
    #N=number of samples;p=probability of sampling a cell
    #generates subsampled trees from the N0 trees at time t
    copy_tree=copy.deepcopy(tree)
    sampled_cells, empty = sample_leaves(copy_tree, p)
    for cell in sampled_cells:
        parent=next(copy_tree.predecessors(cell))
        cousins_we_saw=[cell]
        while parent!='root':
            cousins=getleaves_of_node_digraph(copy_tree,parent)
            for x in cousins_we_saw:
                cousins.remove(x)
            if len(cousins)==0:
                parent=next(copy_tree.predecessors(parent))
                continue
            flag=0
            for i in range(0,len(cousins)):
                for j in range(0,len(sampled_cells)):
                    if cousins[i]==sampled_cells[j]:
                        flag=1
                        cousins_we_saw.append(cousins[i])
                        break

            if flag==1:
                parent=next(copy_tree.predecessors(parent))
            else:
                goodchild=find_child(copy_tree,parent,cell)
                grandparent=next(copy_tree.predecessors(parent))
                copy_tree.add_edge(grandparent,goodchild)
                copy_tree.edges[grandparent,goodchild]['time'] = copy_tree.edges[parent,goodchild]['time']
                copy_tree.remove_edge(grandparent,parent)
                copy_tree.remove_edge(parent,goodchild)

                while getleaves_of_node_digraph(copy_tree,parent) != [parent]:
                    for leaf in getleaves_of_node_digraph(copy_tree,parent):
                        copy_tree.remove_node(leaf)
                copy_tree.remove_node(parent)
                parent=grandparent
    return copy_tree, empty



