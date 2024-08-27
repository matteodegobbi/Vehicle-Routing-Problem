# Vehicle-Routing-Problem

# How to use the code
To run the main genetic algorithm you need to install the following libraries:
* numpy
* matplotlib
* numba
* ORTools
* networkx

The files in the program are:
* VRP.py slow implementation of the genetic algorithm that allows to draw the graphs using networkx
* numbaVRP.py fast version of the algorithm using numba to compile the python code and make it faster in order to tackle bigger instances of CVRP
* ORToolsVRP.py which utilizes the Google ORTools Solver to find a solution to the CVRP using Guided Local Search (most of the code provided by Google with some adaptations)
* GA.ipynb jupyter notebook used to run and visualize the results of the program, the parameters in the notebook can be changed to see the effect on the runtime and the quality of the routes found

At the end of the notebook we offer a comparison on the solutions found by the two different algorithms for the specific instance.

---

We also provide a different implementation of the genetic algorithm done in pure python and using a different crossover operator: partially mapped crossover
