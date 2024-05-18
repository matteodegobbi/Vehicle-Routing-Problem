import numpy as np
from numba import jit
from numba import njit
from numba.experimental import jitclass
from numba import types
from numba.typed import Dict
import random

@njit
def numba_fitness(curr_chromosome,capacities,edge_weights,truck_capacity):
    curr_truck = 0
    total_distance = edge_weights[(-1,curr_chromosome[0])]
    capacity_curr = capacities[curr_chromosome[0]]
    for i in range(1,len(curr_chromosome)):
        capacity_curr += capacities[curr_chromosome[i]]
        #print(capacity_curr.shape)
        if capacity_curr > truck_capacity:
            #ended route
            curr_truck+=1
            total_distance+= (edge_weights[(curr_chromosome[i-1],-1)] + edge_weights[(-1,curr_chromosome[i])])
            capacity_curr = capacities[curr_chromosome[i]]
        else:
            total_distance+= edge_weights[(curr_chromosome[i-1],curr_chromosome[i])]
    total_distance+=edge_weights[(len(curr_chromosome)-1,-1)]
    curr_truck += 1
    return total_distance
@njit
def numba_run(population,capacities,edge_weights,truck_capacity,n_clients,n_iterations : int = 100,decimation_factor :int = 5,mutation_p:float = 0.2):
    kept_population = len(population)//decimation_factor
    fitnesses = np.empty(len(population))
    iteration_best_fitness = np.inf
    iteration_best_chromosome = None
    
    best_fitness = np.inf
    best_chromosome = None

    new_population = np.empty_like(population)
    for iteration in range(n_iterations):
        for pop_index in range(len(population)):
            fitnesses[pop_index] = numba_fitness(population[pop_index],capacities,edge_weights,truck_capacity)
            
        best_chrom_index = np.argmin(fitnesses)
        iteration_best_fitness = fitnesses[best_chrom_index]
        iteration_best_chromosome = population[best_chrom_index].copy()
        probs = (1/fitnesses)/np.sum(1/fitnesses)
        #selected_indices = np.random.choice(np.arange(len(population)), kept_population, p=fitnesses,replace=False)
        temp_range = np.arange(len(population))
        #selected_indices = temp_range[np.searchsorted(np.cumsum(probs), np.random.random(), side="right")]
        selected_indices = numba_choice(temp_range,probs,kept_population)
        population[0:kept_population] = population[selected_indices]
        ###GENERATE NEW POPULATION with EAX
        new_population_index = 0
        while new_population_index < len(population):
            np.random.shuffle(population[:kept_population])
            pop_index = 0
            while pop_index < kept_population-1 and new_population_index < len(population):
                first_child = numba_aex(population,pop_index,pop_index+1,n_clients) 
                new_population[new_population_index] = first_child
                new_population_index+=1
                pop_index+=1
        population[:] = new_population[:] 
        ####MUTATION
        for i in range(len(population)):
            mut = random.random()
            if mut < mutation_p:
                index1 = random.randint(0,n_clients)
                index2 = random.randint(0,n_clients)
                temp = population[index1]
                population[index1] = population[index2]
                population[index2] = temp
        ###ELITISM
        if iteration_best_chromosome is not None:
            population[0] = iteration_best_chromosome.copy()
        if iteration_best_fitness < best_fitness:
            best_fitness = iteration_best_fitness
            best_chromosome = iteration_best_chromosome.copy()
    return best_chromosome, best_fitness, population

@njit
def numba_aex(population,index_chrom1 : int , index_chrom2 : int,n_clients):
    new_chrom = np.empty_like(population[index_chrom1])
    current_gene = population[index_chrom1][0]
    new_chrom[0] = current_gene
    available_genes = set(range(0,n_clients))
    available_genes.remove(current_gene)
    for i in range(1,len(new_chrom)):
        if i % 2 != 0:
            index_chrom = index_chrom1
        else:
            index_chrom = index_chrom2
        candidate_gene_index = ((np.where(population[index_chrom] == current_gene)[0][0])+1)%len(new_chrom)
        #print(candidate_gene_index)
        if population[index_chrom][candidate_gene_index] in available_genes:
            current_gene = population[index_chrom][candidate_gene_index]
        else:
            current_gene = np.random.choice(np.array(list(available_genes)))
        new_chrom[i] = current_gene
        available_genes.remove(current_gene)
    return new_chrom
@njit
def numba_choice(population, weights, k):
    # Get cumulative weights
    wc = np.cumsum(weights)
    # Total of weights
    m = wc[-1]
    # Arrays of sample and sampled indices
    sample = np.empty(k, population.dtype)
    sample_idx = np.full(k, -1, np.int32)
    # Sampling loop
    i = 0
    while i < k:
        # Pick random weight value
        r = m * np.random.rand()
        # Get corresponding index
        idx = np.searchsorted(wc, r, side='right')
        # Check index was not selected before
        # If not using Numba you can just do `np.isin(idx, sample_idx)`
        for j in range(i):
            if sample_idx[j] == idx:
                continue
        # Save sampled value and index
        sample[i] = population[idx]
        sample_idx[i] = population[idx]
        i += 1
    return sample