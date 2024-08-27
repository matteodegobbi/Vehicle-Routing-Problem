import numpy as np
import random
import math
import pop # type: ignore

# Define the locations of the cities and the warehouse as an array of coordinates
cities = np.array([
    [0, 0],     # Warehouse location
    [1125, 48], # City 1
    [311, 145], # City 2
    [245, 1008],# City 3
    [790, 789], # City 4
    [998, 5444] # City 5
])

# Set the maximum capacity of the vehicles
capacity = 10000

# Define the demand at each location (warehouse demand is zero)
demand = np.array([0, 89, 56 ,234, 941, 456])

# Set the number of vehicles available
vehicles = 2

# define the initial population size and mutation rate for the genetic algorithm
population_size = 3
mutation = 0.05

# set the number of iterations (steps) for the simulation
num_steps = 10000

# logarithmic decay parameters for simulated annealing or fitness evaluation adjustment
start = 10
stop = 0.1
logarithmic_steps = np.logspace(np.log(start), np.log(stop), num_steps)
# logarithmic_steps now holds the decay values

# Initialize the population for the genetic algorithm
population = pop.Population(cities, vehicles, population_size, capacity, demand, mutation)

# variables to track generations and evaluation intervals
since_last_eval, i = 0, 0

#run the genetic algorithm until a stopping criterion is met (based on fitness and generations)
while not population.Is_finished or population.generations < 10000:
    #Store the fitness value of the previous generation
    last_fitness = population.fitness
    
    #generatee the next population and calculate its fitness
    population.next_generation()
    population.calculate_fitness()
    
    # periodically evaluate the fitness based on the logarithmic decay
    if since_last_eval > logarithmic_steps[i]: 
        population.evaluate(last_fitness) 
        since_last_eval = 0  
        i += 1
    
    #increment the counter for the next evaluation check
    since_last_eval += 1 
    
print("total generations:", population.generations, "\n")
print("Best Best path:")
population.print_Best()
print("with the shortest distance of:", population.record_distance)

'''
Initialization:

City locations, demands, number of vehicles, population size, and mutation rate are defined.
The population is initialized with the first generation.
A logarithmic decay is calculated for evaluation purposes
Main Loop:

The genetic algorithm runs, generating new populations, calculating fitness, and occasionally evaluating progress.
The loop continues until the population reaches a certain number of generations or a fitness goal.
Evaluation:

Evaluation of fitness is performed based on the logarithmic decay schedule, adjusting the strategy over time
'''