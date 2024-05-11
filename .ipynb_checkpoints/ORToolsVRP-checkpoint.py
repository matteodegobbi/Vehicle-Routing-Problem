import numpy as np
import networkx as nx
import math
import random
#from sklearn.cluster import KMeans
from typing import List
import matplotlib
import colorsys
from VRP import VRP 
from numbaVRP import *
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy
from VRP import draw_ORTools_solution

def print_solution(data, manager, routing, solution):
    routes = []
    """#prints solution on console."""
    #print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        #print(plan_output)
        total_distance += route_distance
        total_load += route_load
        routes.append(route)
    #print(f"Total distance of all routes: {total_distance}m")
    #print(f"Total load of all routes: {total_load}")
    return routes


def solve(data):
    """Solve the CVRP problem."""
    #data = create_data_model()
    #print(np.array(data["distance_matrix"]).shape)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # #print solution on console.
    if solution:
        return print_solution(data, manager, routing, solution)
def get_ORTools_result(vrp : VRP):
    INTEGER_MULTPLIER = 1000
    client_locs = list(vrp.client_locations)
    client_locs.insert(0, np.array(vrp.warehouse_loc))
    distances = []
    unnormalized_distances = []
    for i,(x,y) in enumerate(client_locs):
        distances_i = []
        unnorm_distances_i = []
        for j,(xx,yy) in enumerate(client_locs):
            distances_i.append(int(math.sqrt((xx-x)**2+(yy-y)**2))*INTEGER_MULTPLIER)
            unnorm_distances_i.append((math.sqrt((xx-x)**2+(yy-y)**2)))
        distances.append(distances_i)
        unnormalized_distances.append(unnorm_distances_i)
        data = {}
    data["distance_matrix"] = distances
    data["num_vehicles"] = vrp.n_trucks
    data["depot"] = 0
    demands_including_deposit = list((vrp.capacities*INTEGER_MULTPLIER).astype(int))
    demands_including_deposit.insert(0,0)
    data["demands"] = demands_including_deposit
    data["vehicle_capacities"] = [int(vrp.truck_capacity)*INTEGER_MULTPLIER]*vrp.n_trucks
    routes=solve(data)
    total_distance = 0
    for route in routes:
        distance_route = 0
        capacity_route = 0
        route_with_end = copy.deepcopy(route)
        route_with_end.append(0)
        for client_index in range(len(route_with_end)-1):
            distance_route+= unnormalized_distances[route_with_end[client_index]][route_with_end[client_index+1]]
            if route_with_end[client_index] != 0:
                capacity_route += vrp.capacities[route_with_end[client_index]-1]
        total_distance += distance_route
        #print(capacity_route)
    
    rr = [route[1:] for route in routes]
    rr = [[x-1 for x in r ] for r in rr]#from ortools index to VRP class index
    flat_list = [
        x 
        for xs in rr
        for x in xs
    ]
    #print(flat_list)
    draw_ORTools_solution(vrp,rr)
    return total_distance
