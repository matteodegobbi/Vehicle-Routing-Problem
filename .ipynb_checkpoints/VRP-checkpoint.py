import numpy as np
import networkx as nx
import math
import random
#from sklearn.cluster import KMeans
from typing import List
import matplotlib
import colorsys
import copy
WAREHOUSE_COLOR = 'magenta'
def get_truck_colors(n_trucks : int) -> List[str]:
    cmap = matplotlib.colormaps['Spectral']
    #color_rgba  = cmap(np.linspace(0.0,1.0,n_trucks))
    color_rgba = getDistinctColors(n_trucks)
    
    colors_hex = []
    for c in color_rgba:
        colors_hex.append('#{:02x}{:02x}{:02x}'.format(*(c)))
        #colors_hex.append('#{:02x}{:02x}{:02x}'.format(*(c*255).astype(int)))
    return colors_hex


def HSVToRGB(h, s, v): 
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(n): 
    huePartition = 1.0 / (n + 1) 
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)) 

def draw_ORTools_solution(vrp,routes):
    routes = copy.deepcopy(routes)
    colors = get_truck_colors(len(routes))
    colored_graph = nx.Graph()
    colored_graph.add_node('B',pos=vrp.warehouse_loc,color=WAREHOUSE_COLOR)

    positions = nx.get_node_attributes(vrp.graph,'pos')
    for truck, route in enumerate(routes):
        route.insert(0,'B')
        route.append('B')
        for client_index, client in enumerate(route):
            if client != 'B':
                colored_graph.add_node(client,pos=positions[client],color = colors[truck])
            if client_index != 0:
                colored_graph.add_edge(route[client_index-1],client,color = colors[truck])

    nx.draw(colored_graph,nx.get_node_attributes(colored_graph,'pos'),
            node_color=nx.get_node_attributes(colored_graph,'color').values(),
            edge_color=nx.get_edge_attributes(colored_graph,'color').values()
            ,with_labels=True)
    
class VRP:
    def __init__(self,map_width : float , map_height : float, n_clients : int, n_trucks : int, truck_capacity : float = 50 ,warehouse_loc : tuple[float,float] = None) :
        self.n_clients = n_clients
        self.n_trucks = n_trucks
        self.truck_capacity = truck_capacity
        self.POPULATION_SIZE = 128
        if warehouse_loc is None:
            self.warehouse_loc = (random.random()*map_width,random.random()*map_height)
        else:
            self.warehouse_loc = warehouse_loc
        self.population = np.zeros((self.POPULATION_SIZE,n_clients),dtype=int)
        self.capacities = np.zeros(n_clients, dtype=float)
        #GRAPH
        client_locations_x = np.random.random((n_clients,1))*map_width
        client_locations_y = np.random.random((n_clients,1))*map_height
        self.client_locations = np.hstack([client_locations_x,client_locations_y])
        self.graph = nx.Graph()
        for i,(x,y) in enumerate(self.client_locations):
            self.graph.add_node(i,pos=(x,y))
        self.graph.add_node(-1,pos=self.warehouse_loc)
        
        for i,(x,y) in enumerate(self.client_locations):
            self.graph.add_edge(i,-1,weight = math.sqrt((self.warehouse_loc[0]-x)**2+(self.warehouse_loc[1]-y)**2),color='r')

            
        #TODO: renderlo piu efficiente con n*(n-1)/2
        for i,(x,y) in enumerate(self.client_locations):
            for j,(xx,yy) in enumerate(self.client_locations):
                if i!= j:
                    self.graph.add_edge(i,j,weight = math.sqrt((xx-x)**2+(yy-y)**2))
                    
    def draw_problem(self, show_edges : bool = False):
        raise Exception("change")
        node_color = []
        for node in self.graph:
            node_color.append("yellow" if node == -1 else "coral")
        if show_edges:
            nx.draw(self.graph, nx.get_node_attributes(self.graph, 'pos'), with_labels=True, node_size=150,node_color = node_color)
        else:
            nx.draw_networkx_nodes(self.graph, nx.get_node_attributes(self.graph, 'pos'), node_size=100,node_color = node_color)
            
    def init_random_population(self,init_capacities = True):
        #self.population = np.random.randint(0, self.n_trucks, size=(self.POPULATION_SIZE,self.n_clients))
        array =  np.arange(0,self.n_clients)
        self.population = np.tile(array, (self.POPULATION_SIZE, 1)) 
        #print(self.population)
        #self.population[np.arange(len(self.population))[:,None], np.random.randn(*self.population.shape).argsort(axis=1)]
        self.population = np.apply_along_axis(np.random.permutation, axis=1, arr=self.population)
        if init_capacities:
            self.capacities = np.random.random(self.n_clients) * self.truck_capacity/3
        #print("aaaaaaaaaaaaa")
        #print(self.population)
    def fitness_function(self,index):# da controlalre se e' uguale alla __fast
        #USES A GREEDY APPROACH DESCRIBED IN https://www.researchgate.net/publication/268043232_Comparison_of_eight_evolutionary_crossover_operators_for_the_vehicle_routing_problem 
        curr_chromosome = self.population[index]
        routes = []
        curr_route = ["B",curr_chromosome[0]]
        curr_truck = 0
        total_distance = self.graph.edges[(-1,curr_chromosome[0])]['weight']
        capacity_curr = self.capacities[curr_chromosome[0]]
        for i in range(1,len(curr_chromosome)):
            capacity_curr += self.capacities[curr_chromosome[i]]
            
            #print(capacity_curr)
            if capacity_curr > self.truck_capacity:
                #ended route
                #print(f"Ending route: truck number {curr_truck} filled with {capacity_curr-self.capacities[index][curr_chromosome[i]]} kg")
                #capacity_curr = 0
                curr_truck+=1
                total_distance+=self.graph.edges[(curr_chromosome[i-1],-1)]['weight']
                curr_route.append("B")
                routes.append(curr_route)
                curr_route = ["B",curr_chromosome[i]]
                total_distance+=self.graph.edges[(-1,curr_chromosome[i])]['weight']
                capacity_curr = self.capacities[curr_chromosome[i]]
            else:
                curr_route.append(curr_chromosome[i])
                total_distance+=self.graph.edges[(curr_chromosome[i-1],curr_chromosome[i])]['weight']
        curr_route.append("B")
        routes.append(curr_route)
        total_distance+=self.graph.edges[(len(curr_chromosome)-1,-1)]['weight']
        curr_truck += 1
        #print(f"total distance: {total_distance}")
        #print(f"number of used trucks: {curr_truck}")
        '''if curr_truck > self.n_trucks:
            raise Exception("Numero di camion nella soluzione superiore al consentito")'''
        return total_distance, routes
    def __fast_fitness_function(self,curr_chromosome):
        curr_truck = 0
        total_distance = self.graph.edges[(-1,curr_chromosome[0])]['weight']
        capacity_curr = self.capacities[curr_chromosome[0]]
        for i in range(1,len(curr_chromosome)):
            capacity_curr += self.capacities[curr_chromosome[i]]
            if capacity_curr > self.truck_capacity:
                #ended route
                curr_truck+=1
                total_distance+= (self.graph.edges[(curr_chromosome[i-1],-1)]['weight'] + self.graph.edges[(-1,curr_chromosome[i])]['weight'])
                capacity_curr = self.capacities[curr_chromosome[i]]
            else:
                total_distance+=self.graph.edges[(curr_chromosome[i-1],curr_chromosome[i])]['weight']
        total_distance+=self.graph.edges[(len(curr_chromosome)-1,-1)]['weight']
        curr_truck += 1
        '''if curr_truck > self.n_trucks:
            raise Exception("Numero di camion nella soluzione superiore al consentito")'''
        return total_distance
                
    def draw_chromosome_paths(self,chromosome_index : int = 0):
        _,routes = self.fitness_function(chromosome_index)
        #FORSE ANDREBBE N_TRUCKS AL POSTO DI LEN(ROUTES
        colors = get_truck_colors(len(routes))
        colored_graph = nx.Graph()
        colored_graph.add_node("B",pos=self.warehouse_loc,color=WAREHOUSE_COLOR)

        positions = nx.get_node_attributes(self.graph,'pos')
        for truck, route in enumerate(routes):
            for client_index, client in enumerate(route):
                if client != 'B':
                    colored_graph.add_node(client,pos=positions[client],color = colors[truck])
                if client_index != 0:
                    pass
                    colored_graph.add_edge(route[client_index-1],client,color = colors[truck])
    
        nx.draw(colored_graph,nx.get_node_attributes(colored_graph,'pos'),
                node_color=nx.get_node_attributes(colored_graph,'color').values(),
                edge_color=nx.get_edge_attributes(colored_graph,'color').values()
                ,with_labels=True)
    def alternating_edge_crossover(self,index_chrom1 : int , index_chrom2 : int):
        new_chrom = np.empty_like(self.population[index_chrom1])
        current_gene = self.population[index_chrom1][0]
        new_chrom[0] = current_gene
        available_genes = set(range(0,self.n_clients))
        available_genes.remove(current_gene)
        for i in range(1,len(new_chrom)):
            if i % 2 != 0:
                index_chrom = index_chrom1
            else:
                index_chrom = index_chrom2
            candidate_gene_index = ((np.where(self.population[index_chrom] == current_gene)[0][0])+1)%len(new_chrom)
            #print(candidate_gene_index)
            if self.population[index_chrom][candidate_gene_index] in available_genes:
                current_gene = self.population[index_chrom][candidate_gene_index]
            else:
                current_gene = random.choice(list(available_genes))
            new_chrom[i] = current_gene
            available_genes.remove(current_gene)
        return new_chrom

    def run(self,n_iterations : int = 100,decimation_factor :int = 5, mutation_p = 0.2):
        kept_population = len(self.population)//decimation_factor
        fitnesses = np.empty(len(self.population))
        iteration_best_fitness = float('inf')
        iteration_best_chromosome = None
        
        best_fitness = float('inf')
        best_chromosome = None

        new_population = np.empty_like(self.population)
        for iteration in range(n_iterations):
            fitnesses = np.apply_along_axis(self.__fast_fitness_function,1,self.population)
            best_chrom_index = np.argmin(fitnesses)
            iteration_best_fitness = fitnesses[best_chrom_index]
            iteration_best_chromosome = self.population[best_chrom_index].copy()
            #list(range(len(self.population)))
            selected_indices = random.choices(np.arange(len(self.population)), weights=fitnesses, k=kept_population)
            self.population[0:kept_population] = self.population[selected_indices]
            ###GENERATE NEW POPULATION with EAX
            new_population_index = 0
            while new_population_index < len(self.population):
                np.random.shuffle(self.population[:kept_population])
                pop_index = 0
                while pop_index < kept_population-1 and new_population_index < len(self.population):
                    first_child = self.alternating_edge_crossover(pop_index,pop_index+1) 
                    new_population[new_population_index] = first_child
                    new_population_index+=1
                    pop_index+=1
            self.population[:] = new_population[:] 
            ####MUTATION
            for i in range(len(self.population)):
                mut = random.random()
                if mut < mutation_p:
                    index1 = random.randint(0,self.n_clients)
                    index2 = random.randint(0,self.n_clients)
                    temp = self.population[index1]
                    self.population[index1] = self.population[index2]
                    self.population[index2] = temp
            ###ELITISM
            if iteration_best_chromosome is not None:
                self.population[0] = iteration_best_chromosome.copy()
            if iteration_best_fitness < best_fitness:
                best_fitness = iteration_best_fitness
                best_chromosome = iteration_best_chromosome.copy()
        return best_chromosome, best_fitness