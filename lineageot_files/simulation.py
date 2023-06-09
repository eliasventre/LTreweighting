# Main file for creating simulated data

import numpy as np
import math
import autograd
import warnings
import copy


class SimulationParameters:
    """
    Storing the parameters for simulated data
    """
    
    def __init__(self,
                 timestep = 0.05,                              # Time step for Euler integration of SDE
                 diffusion_constant = .001,                    # Diffusion constant
                 mean_division_time = 10,                      # Meantime before cell division
                 division_time_distribution = "normal",        # Distribution of cell division times
                 mean_death_time = 1e16,                       # Meantime before cell death
                 division_time_std = 0,                        # Standard deviation of division times
                 death_variability = 2,                        # Death variability to test
                 min_death_time = 1e16,                           # min death time
                 min_division_time = 1e16,                           # min birth tiem
                 target_num_cells = np.inf,                    # Upper bound on number of observed cells
                 mutation_rate = 1,                            # Rate at which barcodes mutate
                 flow_type = 'bifurcation',                    # Type of flow field simulated
                 non_constant_branching_rate = False,          # Constant or not branching rate
                 non_constant_death_rate = False,              # Constant or not death rate
                 x0_speed = 1,                                 # Speed at which cells go through transition
                 barcode_length = 15,                          # Number of elements of the barcode
                 back_mutations = False,                       # Whether barcode elements can mutate multiple times
                 num_genes = 3,                                # Number of genes defining cell state
                 initial_distribution_std = 0,                 # Standard deviation of initial cell distribution
                 alphabet_size = 200,                          # Number of possible mutations for a single barcode element
                 relative_mutation_likelihoods = np.ones(200), # Relative likelihood of each mutation
                 keep_tree = True,                             # Whether simulation output includes tree structure
                 enforce_barcode_reproducibility = True,       # Whether to use reproducible_poisson sampling
                 keep_cell_seeds = True                        # Whether to store seeds to reproduce cell trajectories individually
                 ):

        self.timestep = timestep
        self.diffusion_constant = diffusion_constant
        if flow_type in {'mismatched_clusters', 'convergent', 'partial_convergent'}:
            self.diffusion_matrix = np.diag(np.sqrt(diffusion_constant)*np.ones(num_genes))
            self.diffusion_matrix[0, 0] = 0.005*np.sqrt(x0_speed) #small diffusion in 'time' dimension
        else:
            self.diffusion_matrix = np.sqrt(diffusion_constant)

        self.non_constant_branching_rate = non_constant_branching_rate
        self.non_constant_death_rate = non_constant_death_rate
        self.mean_division_time = mean_division_time
        self.mean_death_time = mean_death_time
        self.min_division_time = min_division_time
        self.min_death_time = min_death_time
        self.division_time_distribution = division_time_distribution
        self.division_time_std = division_time_std
        self.death_variability = death_variability
        self.target_num_cells = target_num_cells

        self.mutation_rate = mutation_rate
        self.flow_type = flow_type
        self.x0_speed = x0_speed
        
        self.barcode_length = barcode_length
        self.back_mutations = back_mutations
        self.num_genes = num_genes
        self.initial_distribution_std = initial_distribution_std
        self.alphabet_size = alphabet_size
        self.keep_tree = keep_tree
        self.enforce_barcode_reproducibility = enforce_barcode_reproducibility
        self.keep_cell_seeds = keep_cell_seeds

        if len(relative_mutation_likelihoods) > alphabet_size:
            warnings.warn('relative_mutation_likelihoods too long: ignoring extra entries')
        elif len(relative_mutation_likelihoods) < alphabet_size:
            raise ValueError('relative_mutation_likelihods must be as long as alphabet_size')
        
        self.relative_mutation_likelihoods = relative_mutation_likelihoods[0:alphabet_size]
        if not self.back_mutations:
            self.relative_mutation_likelihoods[0] = 0

        self.mutation_likelihoods = relative_mutation_likelihoods/sum(relative_mutation_likelihoods)


class Cell:
    """
    Wrapper for (rna expression, barcode) arrays
    """
    def __init__(self, x, barcode, seed = None):
        self.x = x
        self.barcode = barcode

        # Note: to recover a cell's trajectory you
        # need both the seed and initial condition
        #
        # This only stores the seed
        if seed != None:
            self.seed = seed
        else:
            self.seed = np.random.randint(42)
        return

    def reset_seed(self):
        self.seed = np.random.randint(42)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return '<Cell(%s, %s)>' % (self.x, self.barcode)
    def __str__(self):
        return 'Cell with x: %s\tBarcode: %s' % ( self.x, self.barcode)


def MFL_flow_2(x, t):
    dim = np.size(x)
    x0 = np.array([1.5, ] + [0, ]*(dim - 1))
    x1 = -np.array([1.5, ] + [0, ]*(dim - 1))
    res = np.zeros(dim)
    for i in range(dim):
        res[i] -= ((x - x0) * (x - x1)**2 + (x - x0)**2 * (x - x1))[0] * (i == 0) + 20*(t+x[1]) * (i == 1) + 20*x[2] * (i > 1)
    return res

def MFL_flow_2_F4(x, t):
    dim = np.size(x)
    x0 = np.array([1.5, ] + [0, ]*(dim - 1))
    x1 = -np.array([1.5, ] + [0, ]*(dim - 1))
    res = np.zeros(dim)
    for i in range(dim):
        res[i] -= ((x - x0) * (x - x1)**2 + (x - x0)**2 * (x - x1))[0] * (i == 0) + 2*(t+x[1]) * (i == 1) + 20*x[2] * (i > 1)
    return res

def MFL_flow_3(x, t):
    dim = np.size(x)
    x0 = np.array([1.5, ] + [0, ]*(dim - 1))
    x1 = -np.array([1.5, ] + [0, ]*(dim - 1))
    x2 = np.array([0, ] + [0, ]*(dim - 1))
    res = np.zeros(dim)
    for i in range(dim):
        res[i] -= (2 * (x - x0) * (x - x1)**2 * (x - x2)**2 + 2 * (x - x0)**2 * (x - x1) * (x - x2)**2 +
                   (x - x0)**2 * (x - x1)**2 * (x - x2) / 9)[0] * (i == 0) \
                  + 20*(x[1] + t) * (i == 1) + 20*x[i] * (i > 1)
    return res

def MFL_flow_4(x, t, diff):
    dim = np.size(x)
    res = np.zeros(dim)
    for i in range(dim):
        res[i] += 5*diff*(np.sin(2*x[0]/diff) + (1 - 2*(x[0] < 0))) * (i == 0) * (x[1] < -.5)  \
                  - 20*(x[1] + t) * (i == 1) - 20*x[i] * (i > 1)
    return res


def vector_field(x, t, params):
    """
    Selects a vector field and returns its value at x
    """
    if params.flow_type == "MFL_2":
        return MFL_flow_2(x, t)
    if params.flow_type == "MFL_2_F4":
        return MFL_flow_2_F4(x, t)
    elif params.flow_type == "MFL_3":
        return MFL_flow_3(x, t)
    elif params.flow_type == "MFL_4":
        return MFL_flow_4(x, t, params.diffusion_constant)
    else:
        warnings.warn("Unknown vector field option. Vector field set to zero.")
        return np.zeros(x.shape)




def center(barcode, params):
    """
    Returns the center of the distribution p(x0|barcode)
    """
    c = np.zeros(params.num_genes)
    l = min(params.num_genes, params.barcode_length)
    c[0:l] = barcode[0:l]
    return c


def sample_barcode(params):
    """
    Samples an initial barcode
    """
    if params.back_mutations:
        barcode = np.random.randint(0,params.alphabet_size,params.barcode_length)
    else:
        barcode = np.zeros(params.barcode_length)
    return barcode

def sample_x0(barcode, params):
    """
    Samples the initial position in gene expression space
    """
    return center(barcode, params) + params.initial_distribution_std*np.random.randn(params.num_genes)

def sample_cell(params):
    """
    Samples an initial cell
    """
    barcode = sample_barcode(params)
    x0 = sample_x0(barcode, params)
    return Cell(x0, barcode)


def evolve_x(initial_x, time_init, time, params):
    """
    Returns a sample from Langevin dynamics following potential_gradient
    """
    current_x = initial_x
    if params.flow_type != None:
        current_time = 0
        #print("Evolving x for time %f\n"% time)
        while current_time < time:
            current_x = (current_x 
                         + params.timestep*vector_field(current_x, time_init + current_time - time, params)
                         + np.sqrt(params.timestep)*np.dot(params.diffusion_matrix,
                                                           np.random.randn(initial_x.size)))
            current_time = current_time + params.timestep
        #print("Finished evolving x\n")
    return current_x

def evolve_x_fromx0(initial_x, time_init, time, params):
    """
    Returns a sample from Langevin dynamics following potential_gradient
    """
    current_x = initial_x
    if params.flow_type != None:
        current_time = 0
        #print("Evolving x for time %f\n"% time)
        while current_time < time:
            current_x = (current_x
                         + params.timestep*vector_field(current_x, time_init + current_time, params)
                         + np.sqrt(params.timestep)*np.dot(params.diffusion_matrix,
                                                           np.random.randn(initial_x.size)))
            current_time = current_time + params.timestep
        #print("Finished evolving x\n")
    return current_x


def func_mean_division_time(x, params):
    if params.flow_type == "MFL_2": return params.mean_division_time / ((np.tanh(2*x[:, 0]) + 1)/1.5)
    if params.flow_type == "MFL_2_F4": return params.mean_division_time / ((np.tanh(2*x[:, 0]) + 1)/1.5)
    elif params.flow_type == "MFL_3": return params.mean_division_time / ((np.tanh(2*x[:, 0]) + 1)/1.2)
    elif params.flow_type == "MFL_4":
        return params.mean_division_time * (.3*(x[:, 1] > -.5) +
                (1 / (np.tanh(x[:, 0]/(3*params.diffusion_constant)) + 1)/1.5) * (x[:, 1] <= -.5)
                * (1.8 * (x[:, 0] <= 5*params.diffusion_constant) + 1.5 * (x[:, 0] > 5*params.diffusion_constant)))
    else: return params.mean_division_time


def func_mean_death_time(x, params):
    if params.flow_type == "MFL_2": return params.mean_death_time * (2 * (x[:, 0] <= 0) + 4 * (x[:, 0] > 0))
    elif params.flow_type == "MFL_3": return params.mean_death_time * \
                                           (.6 + 5*params.death_variability/6 + params.death_variability * (1 * (x[:, 0] > .5) + (-1) * (x[:, 0] < -.5)))
    elif params.flow_type == "MFL_4":
        return params.mean_division_time * params.mean_death_time * (2 * (x[:, 0] <= 0) + 4 * (x[:, 0] > 0))
    else: return params.mean_death_time


def sample_division_time(x, params):
    """
    Samples the time until a cell divides
    """
    if not params.non_constant_branching_rate:
        t = np.random.exponential(params.mean_division_time)
    else:
        x_arr = x.x.reshape((1, np.size(x.x)))
        t = np.random.exponential(func_mean_division_time(x_arr, params)[0])

    return t

def sample_death_time(x, params):
    """
    Samples the time until a cell divides
    """
    if not params.non_constant_death_rate:
        t = np.random.exponential(params.mean_death_time)

    else:
        x_arr = x.x.reshape((1, np.size(x.x)))
        t = np.random.exponential(func_mean_death_time(x_arr, params)[0])
    return t

def sample_next_time(x, params):
    """
    Samples the time until a cell divides
    """
    if not params.non_constant_death_rate and not params.non_constant_death_rate:
        t = 1e16
    else:
        t = np.random.exponential(min(params.min_death_time, params.min_division_time))
    return t


def evolve_b(initial_barcode, time, params):
    """
    Returns the new barcode after mutations have occurred for some time
    """
    # Setting the seed for reproducibility, specifically to make the
    # process consistent with different time intervals.
    # If the random seed has been set before evolve_b is called, then
    # we want the mutations that occur in time t1 to be a subset of the
    # mutations that occur in time t2 if t2 > t1.

    mutation_seed = np.random.randint(42)

    rate = time*params.mutation_rate
    if (rate > 10) & params.enforce_barcode_reproducibility:
        warnings.warn("Large number of mutations expected on one edge. " +
                      "Falling back on slow non-numpy Poisson sampling.")
        num_mutations = reproducible_poisson(rate)
    else:
        num_mutations = np.random.poisson(rate)

    # need to reset the seed after the Poisson sample because the state of the
    # RNG after Poisson sampling depends on the parameter of the Poisson distribution
    np.random.seed(mutation_seed)
    for m in range(num_mutations):
        mutate_barcode(initial_barcode, params)
    return initial_barcode

def mutate_barcode(barcode, params):
    """
    Randomly changes one entry of the barcode
    """
    changed_entry = np.random.randint(0, len(barcode)) # not modeling different mutation rates across cut sites
    if params.back_mutations | (barcode[changed_entry] == 0):
        # if no back mutations, only change unmutated sites
        barcode[changed_entry] = np.random.choice(range(params.alphabet_size), p = params.mutation_likelihoods)
    return barcode

def evolve_cell(initial_cell, time_init, time, params):
    """
    Returns a new cell after both barcode and x have evolved for some time
    """
    # np.random.seed(initial_cell.seed) # allows reproducibility of individual trajectories
    # new_b = evolve_b(initial_cell.barcode, time, params)

    # np.random.seed(initial_cell.seed)
    new_x = evolve_x(initial_cell.x, time_init, time, params)

    if params.keep_cell_seeds:
        return Cell(new_x, initial_cell.barcode, seed = initial_cell.seed)
    else:
        return Cell(new_x, initial_cell.barcode)



def mask_barcode(barcode, p):
    """
    Replaces a subset of the entries of barcode with -1 to simulate missing data

    Entries are masked independently with probability p 

    Also works for an array of barcodes
    """
    mask = np.random.rand(*barcode.shape) < p
    barcode[mask] = -1
    return barcode




def sample_descendants(initial_cell, time_init, time, params, target_num_cells = None):
    """
    Samples the descendants of an initial cell
    """
    global num_cells
    if time_init == time: num_cells = 1
    assert(isinstance(initial_cell, Cell))
    if target_num_cells == None:
        target_num_cells = params.target_num_cells
    next_birth_time = sample_division_time(initial_cell, params)
    next_death_time = sample_death_time(initial_cell, params) + 1e16 * (num_cells == 1)
    next_time = sample_next_time(initial_cell, params)
    while next_time < min(next_birth_time, next_death_time):
        initial_cell = evolve_cell(initial_cell, time_init, next_time, params)
        next_time = sample_next_time(initial_cell, params)
        next_birth_time = sample_division_time(initial_cell, params)
        next_death_time = sample_death_time(initial_cell, params) + 1e16 * (num_cells == 1)
    next_division_time = min(next_birth_time, next_death_time)

    if (next_division_time > time) | (target_num_cells == 1):
        #print("In final division stage\n")
        if params.keep_tree:
            return[(evolve_cell(initial_cell, time_init, time, params), time)]
        else:
            return [evolve_cell(initial_cell, time_init, time, params)]
    else:
        time_remaining = time - next_division_time
        if next_division_time == next_birth_time:
            num_cells += 1
            target_num_cells_1, target_num_cells_2 = split_targets_between_daughters(time_remaining, target_num_cells, params)
            while (target_num_cells_1 == 0) | (target_num_cells_2 == 0):
                # wait until we get a division where both daughters have observed descendants
                t = sample_division_time(initial_cell, params)
                if t > time_remaining:
                    # In rare cases, you can have:
                    # 1) time_remaining greater than mean_division_time
                    # 2) split_targets assigning 0 cells to one branch, so we skip to the next division and
                    # 3) sample_division_time greater than time_remaining
                    # leading to negative time_remaining and an error
                    #
                    # If mean_division_time >> division_time_std, this will only happen at the last division
                    # which is the case this handles
                    if target_num_cells == 2:
                        # In this case, since the branch we chose doesn't divide before the end of the simulation,
                        # we can assign one of its samples to the other branch
                        target_num_cells_1 = 1
                        target_num_cells_2 = 1
                        break
                    else:
                        raise(ValueError)
                else:
                    next_division_time = next_division_time + t
                    time_remaining = time - next_division_time
                    target_num_cells_1, target_num_cells_2 = split_targets_between_daughters(time_remaining, target_num_cells, params)
            cell_at_division = evolve_cell(initial_cell, time_init, next_division_time, params)
            assert(cell_at_division.seed >= 0)
            daughter_1 = cell_at_division.deepcopy()
            daughter_2 = cell_at_division.deepcopy()
            daughter_1.reset_seed()
            daughter_2.reset_seed()
            if params.keep_tree:
                # store as list of lists that keeps tree structure, ancestors, and ancestor times
                return [sample_descendants(daughter_1, time_init, time_remaining, params, target_num_cells_1),
                         sample_descendants(daughter_2, time_init, time_remaining, params, target_num_cells_2),
                         (cell_at_division, next_division_time)]
            else:
                # store all cells as a flat list, ignoring ancestors
                return (sample_descendants(daughter_1, time_init, time_remaining, params, target_num_cells_1) +
                        sample_descendants(daughter_2, time_init, time_remaining, params, target_num_cells_2))

        else:
            num_cells -= 1
            if params.keep_tree:
                return None
            else:
                return None

    return


def split_targets_between_daughters(time_remaining, target_num_cells, params):
    """
    Given a target number of cells to sample, divides the samples between daughters
    assuming both have the expected number of descendants at the sampling time
    """
    num_future_generations = np.floor(time_remaining/params.mean_division_time)
    num_descendants = 2**num_future_generations
    
    if target_num_cells > 2*num_descendants:
        # we expect to sample all the cells
        target_num_cells_1 = np.floor(target_num_cells/2)
    else:
        target_num_cells_1 = np.random.hypergeometric(num_descendants, num_descendants, target_num_cells)

    if target_num_cells == np.inf:
        target_num_cells_2 = np.inf
    else:
        target_num_cells_2 = target_num_cells - target_num_cells_1

    return target_num_cells_1, target_num_cells_2



def sample_population_descendants(pop, time, params):
    """
    Samples the descendants of each cell in a population
    pop: list of (expression, barcode) tuples
    """
    sampled_population = []
    num_descendants = np.zeros((len(pop))) # the number of descendants of each cell
    for cell in range(len(pop)):
        descendants = sample_descendants(pop[cell], time, params)
        num_descendants[cell] = len(descendants)
        sampled_population = sampled_population + descendants

    return sampled_population, num_descendants

def flatten_list_of_lists(tree_data):
    """
    Converts a dataset of cells with their ancestral tree structure to a list of cells
    (with ancestor and time information dropped)
    """
    assert isinstance(tree_data, list)
    if len(tree_data) == 1:
        assert(isinstance(tree_data[0][0], Cell))
        return [tree_data[0][0]]
    else:
        assert len(tree_data) == 3 # assuming this is a binary tree
        # the third entry of tree_data should be a (cell, time) tuple
        assert isinstance(tree_data[2][0], Cell)
        return flatten_list_of_lists(tree_data[0]) + flatten_list_of_lists(tree_data[1])
    return

def convert_data_to_arrays(data):
    """
    Converts a list of cells to two ndarrays,
    one for expression and one for barcodes
    """
    expressions = np.array([cell.x for cell in data])
    barcodes = np.array([cell.barcode for cell in data])
    return expressions, barcodes



def sample_pop(num_initial_cells, time, params):
    """
    Samples a population after some intervening time

    num_initial_cells:                          Number of cells in the population at time 0
    time:                                       Time when population is measured
    params:                                     Simulation parameters
    """

    if num_initial_cells == 0:
        return np.zeros((0, params.num_genes)), np.zeros((0, params.barcode_length)), np.zeros((0)), []
    
    initial_population = []
    for cell in range(num_initial_cells):
        initial_population = initial_population + [sample_cell(params)]

    sampled_population, num_descendants = sample_population_descendants(initial_population, time, params)

    expressions, barcodes = convert_data_to_arrays(sampled_population)
    return (expressions, barcodes, num_descendants, initial_population)




def subsample_list(sample, target_num_cells):
    """
    Randomly samples target_num_cells from the sample
    
    If there are fewer than target_num_cells in the sample,
    returns the whole sample
    """

    if target_num_cells > len(sample):
        return sample
    else:
        # not using permutation so the order of elements in sample is preserved
        r = np.random.rand(len(sample))
        sorted_indices = np.argsort(r)
        min_dropped_r = r[sorted_indices[target_num_cells]]
        return sample[r < min_dropped_r]





def subsample_pop(sample, target_num_cells, params, num_cells = None):
    """
    Randomly samples target_num_cells from the sample. Subsampling during the simulation by
    setting params.target_num_cells is a more efficient approximation of this.
    
    If there are fewer than target_num_cells in the sample,
    returns the whole sample

    sample should be either:

    - a list of cells, if params.keep_tree is False
    - nested lists of lists of cells encoding the tree structure, if params.keep_tree is True

    (i.e., it should match the output of sample_descendants with the same params)
    """

    if target_num_cells == 0:
        return []
    elif params.keep_tree:
        if num_cells == None:
            sample_list = flatten_list_of_lists(sample)
            num_cells = len(sample_list)

        if num_cells <= target_num_cells:
            return sample
        else:
            daughter_1_subtree = sample[0]
            daughter_2_subtree = sample[1]
            
            # TODO: there is a lot of redundant flattening happening. Should be
            #       redone if it is slow
            num_cells_1 = len(flatten_list_of_lists(daughter_1_subtree))
            num_cells_2 = len(flatten_list_of_lists(daughter_2_subtree))

            target_num_cells_1 = np.random.hypergeometric(num_cells_1, num_cells_2, target_num_cells)
            target_num_cells_2 = target_num_cells - target_num_cells_1

            # If one subtree does not get sampled, return only the other subtree subsampled
            # with its root 'time_to_parent' adjusted
            if target_num_cells_1 == 0:
                daughter_2_subtree[-1] = (daughter_2_subtree[-1][0], daughter_2_subtree[-1][1] + sample[2][1])
                return subsample_pop(daughter_2_subtree, target_num_cells_2, params, num_cells = num_cells_2)
            elif target_num_cells_2 == 0:
                daughter_1_subtree[-1] = (daughter_1_subtree[-1][0], daughter_1_subtree[-1][1] + sample[2][1])
                return subsample_pop(daughter_1_subtree, target_num_cells_1, params, num_cells = num_cells_1)
            else:
                return [subsample_pop(daughter_1_subtree, target_num_cells_1, params, num_cells = num_cells_1),
                        subsample_pop(daughter_2_subtree, target_num_cells_2, params, num_cells = num_cells_2),
                        sample[2]]

            

    else:
        return subsample_list(sample, target_num_cells)


def reproducible_poisson(rate):
    """
    Samples a single Poisson random variable, in a way
    that is reproducible, i.e. after
    
    np.random.seed(s)
    a = divisible_poisson(r1)
    np.random.seed(s)
    b = divisible_poisson(r2)
    
    with r1 > r2, b ~ binomial(n = a, p = r2/r1)


    This is the standard numpy Poisson sampling algorithm for rate <= 10.
    
    Note that this is relatively slow, running in O(rate) time.
    """
    T = np.exp(-rate)
    k = -1
    t = 1
    while t >= T:
        t = t*np.random.rand()
        k = k+1
        
    return k

