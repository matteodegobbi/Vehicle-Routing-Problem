import numpy as np 
import random
import math

class Population:
    def __init__(self, cities, vehicles, population_size, capacity, demand, mutation):
        # Initialize population with empty arrays for each vehicle in each organism
        self.population = np.empty((population_size, vehicles), dtype=object)
        
        # Optionally initialize the mating pool (commented out for now)
        self.mating_pool = np.empty_like(self.population)
        
        # Initialize various population attributes
        self.generations = 0 
        self.fitness = np.zeros(population_size)
        self.fitness_norm = np.empty_like(self.fitness)
        self.Is_finished = False 
        
        # Track the best solution found so far
        self.best_ever = np.empty((vehicles,), dtype=object) 
        self.record_distance = float("inf")
        
        # Store cities, number of vehicles, population size, vehicle capacity, and demand
        self.total_cities = len(cities) - 1  # exclude the warehouse from total cities count
        self.cities = np.copy(cities)
        self.vehicles = vehicles
        self.population_size = population_size
        self.capacity = capacity
        self.demand = np.copy(demand)
        
        # Set mutation rate based on the number of cities
        self.mutation_rate = mutation * self.total_cities
        
        # Initialize population and calculate distances between cities
        self.initialiaz_population()
        self.distances = self.distances()
        
        # Calculate the fitness for the initial population
        self.calculate_fitness()
        
    def initialiaz_population(self):
        # Generate initial population with random routes for each vehicle
        for i in range(self.population_size):
            rand_cities = np.random.permutation(np.arange(1, self.total_cities + 1))
            
            # Randomly split cities among vehicles
            indici = np.random.choice(self.total_cities, size=self.vehicles - 1, replace=False)
            indici = np.sort(indici)
            indici = np.concatenate([[0], indici, [self.total_cities]])
            
            # Assign routes to vehicles
            route_vehicle = [rand_cities[indici[i]:indici[i+1]] for i in range(self.vehicles)]
            self.population[i] = route_vehicle

    def distances(self):
        # Calculate the Euclidean distance between every pair of cities
        distances = np.zeros((self.total_cities + 1, self.total_cities + 1))
        for i in range(self.total_cities + 1):
            for j in range(self.total_cities + 1):
                if i != j:
                    distances[i][j] = math.sqrt(pow(self.cities[i, 0] - self.cities[j, 0], 2) +
                                                pow(self.cities[i, 1] - self.cities[j, 1], 2))
        return distances
                                    
    def calc_distance(self, organism):
        # Calculate the total distance for a given organism and check if the demand is within vehicle capacity
        total_dist_vehicle_con_wh = 0
        check_demand = True
        for i, path in enumerate(organism):
            dist_vehicle_senza_wh, dist_vehicle_con_warehouse, demand_for_vehicle = 0, 0, 0
            if path.size != 0:
                for j in range(len(path) - 1):
                    dist_vehicle_senza_wh += self.distances[path[j]][path[j+1]]
                    demand_for_vehicle += self.demand[path[j]]
                    
                # Add demand of the last city in the path
                demand_for_vehicle += self.demand[path[-1]]
                if demand_for_vehicle > self.capacity: 
                    check_demand = False
                
                # Calculate total distance including return to the warehouse
                dist_vehicle_con_warehouse = self.distances[path[0]][0] + dist_vehicle_senza_wh + self.distances[path[-1]][0]  
                total_dist_vehicle_con_wh += dist_vehicle_con_warehouse
                
        return total_dist_vehicle_con_wh, check_demand

    def calculate_fitness(self):
        # Evaluate the fitness of each organism based on its total distance and demand constraint
        for i, organism in enumerate(self.population):
            total_dist_for_fit, check_demand = self.calc_distance(organism)
            if not check_demand: 
                self.fitness[i] = 0  # Assign zero fitness if demand exceeds capacity                                  
            else:
                # Calculate fitness, inversely proportional to distance
                self.fitness[i] = 1 / ((10**-7) * total_dist_for_fit + 1)
                
            # Track the best distance and corresponding solution
            if total_dist_for_fit < self.record_distance:
                self.record_distance = total_dist_for_fit
                self.best_ever = self.population[i].copy()

    def evaluate(self, last_fitness):
        # Check if the improvement in fitness is small enough to stop the algorithm
        max_last = np.max(last_fitness)
        max_curr = np.max(self.fitness)
        for i, organism in enumerate(self.population):
            fraz = max_curr / max_last 
            if abs(1 - fraz) < 0.05:
                self.Is_finished = True
                break  
        
    def mutate(self, organism):
        #apply mutations to an organism by swapping genes within the path
        
        #iterate over a number of mutations based on the mutation rate
        for _ in range(int(self.mutation_rate)):
            #loop through each vehicle's path in the organism
            for i, path in enumerate(organism):
                if path.size != 0:  # Ensure the path is not empty
                    #select a random index in the path
                    idx = random.randint(0, len(path) - 1)
                    #get the gene at the selected index
                    gene_to_mutate = organism[i][idx]
                    #find possible genes to swap with (excluding the current gene)
                    possible_genes = [gene for gene in path if gene != gene_to_mutate]
                    #randomly select one of the possible genes
                    new_gene = random.choice(possible_genes)
                    #eplace the gene at the selected index with the new gene
                    organism[i][idx] = new_gene

            
    def partially_mapped_crossover(self, parent1, parent2):
        # Perform partially mapped crossover (PMX) between two parent organisms
        
        # Flatten the parents' routes for easier crossover operation
        flattened_parent1 = [item for sublist in parent1 for item in sublist]
        flattened_parent2 = [item for sublist in parent2 for item in sublist]

        # Initialize children with placeholders (-1)
        size = len(flattened_parent1)
        child1 = [-1] * size
        child2 = [-1] * size

        # Select crossover points randomly
        point1 = random.randint(0, size - 1)
        point2 = random.randint(0, size - 1)
        start = min(point1, point2)  # Ensure start is less than end
        end = max(point1, point2)

        # Copy crossover segments from parents to children
        child1[start:end] = flattened_parent1[start:end]
        child2[start:end] = flattened_parent2[start:end]

        # Map the remaining genes from parent2 to child1
        for i in range(start, end):
            if flattened_parent2[i] not in child1:
                # Find the corresponding gene from parent1 to map to child1
                idx = flattened_parent2.index(flattened_parent1[i])
                while child1[idx] != -1:  # Resolve conflicts
                    idx = flattened_parent2.index(flattened_parent1[idx])
                child1[idx] = flattened_parent2[i]

        # Map the remaining genes from parent1 to child2
        for i in range(start, end):
            if flattened_parent1[i] not in child2:
                # Find the corresponding gene from parent2 to map to child2
                idx = flattened_parent1.index(flattened_parent2[i])
                while child2[idx] != -1:  # Resolve conflicts
                    idx = flattened_parent1.index(flattened_parent2[idx])
                child2[idx] = flattened_parent1[i]

        # Fill in the remaining positions in child1 and child2
        for i in range(size):
            if child1[i] == -1:
                child1[i] = flattened_parent2[i]
            if child2[i] == -1:
                child2[i] = flattened_parent1[i]

        # Restore the original structure of the children from the flattened arrays
        parent1_restored = []
        start_index = 0
        for sublist in parent1:
            sublist_length = len(sublist)
            parent1_restored.append(child1[start_index:start_index + sublist_length])
            start_index += sublist_length

        parent2_restored = []
        start_index = 0
        for sublist in parent2:
            sublist_length = len(sublist)
            parent2_restored.append(child2[start_index:start_index + sublist_length])
            start_index += sublist_length
            
        return parent1_restored, parent2_restored


    def normalize_fitness(self):
        #normalize the fitness values for selection probability calculation
        total_fitness = np.sum(self.fitness)
        self.fitness_norm = self.fitness / total_fitness
       
    def next_generation(self):
        # generate the next generation of the population
        self.generations += 1
        new_population = np.empty_like(self.population)
        
        # Normalize fitness values for selection
        self.normalize_fitness()
        for i in range(0, len(self.population)-1, 2): 
            # Select parents based on normalized fitness
            parent_a = self.pick_one(self.population, self.fitness_norm) 
            parent_b = self.pick_one(self.population, self.fitness_norm)
            
            # Perform crossover and mutation to produce offspring
            son1, son2 = self.partially_mapped_crossover(parent_a, parent_b)
            self.mutate(son1)
            self.mutate(son2)
            
        #Add offspring to the new population
            new_population[i] = son1
            new_population[i+1] = son2

        population = new_population

    def pick_one(self, population, prob):
        #select one organism from the population based on a probability distribution
        index = np.random.choice(len(population), p=prob)
        return population[index].copy()

    def print_Best(self):
        # Print the best solution found so far
        best_solution = self.get_Best()
        for i, paths in enumerate(best_solution):
            print(f"Path {i+1}:", paths)
            
    def get_Best(self):
        # Retrieve the best organism from the current population based on fitness
        max_fitness_index = np.argmax(self.fitness) 
        return self.population[max_fitness_index]


'''

Explanation of Partially Mapped Crossover (PMX)
Partially Mapped Crossover (PMX) is a genetic algorithm crossover operator specifically designed for permutation-based problems.
It combines parts of two parent solutions to create offspring while maintaining the validity of the solution. 
PMX works as follows:

1.Selection of Crossover Points: Two crossover points are chosen randomly within the permutation.
2.Segment Copying: The segments between these points are copied from each parent to the corresponding offspring.
3.Mapping and Replacement: The remaining genes in the offspring are filled in using a mapping derived from the parents. This ensures that no gene is duplicated and all genes are included.


Detailed Comments:
1.Flatten Parents: Convert the nested list structure of each parent into a single list. This simplifies the crossover operation as we deal with linear arrays instead of nested lists

2.Initialize Children: Create empty lists for the children, initially filled with -1, which will later be replaced with genes from the parents

3.Select Crossover Points: Randomly choose two points in the flattened arrays

4.Copy Crossover Segments: Copy the segments between the crossover points from each parent to the respective child. This segment retains the order of genes from the parents.

5.Map Remaining Genes (PMX Logic):
From Parent2 to Child1: For genes outside the crossover segment in child1, use a mapping derived from parent2 to fill in the remaining positions
From Parent1 to Child2: Similarly, fill in the remaining positions in child2 using the mapping derived from parent1

6.Fill Remaining Positions: Any positions in the children that are still -1 are filled with the remaining genes from the opposite parent

7.Restore Original Structure: Convert the flat children lists back into the original nested list structure, matching the format of the parent solutions.

'''