# 0-1 Knapsack Problem with Artificial Bee Colony
This project implements a solution to the 0-1 knapsack problem using the Artificial Bee Colony (ABC) optimization algorithm.
# Problem Definition
The 0-1 knapsack problem is a combinatorial optimization problem where we aim to maximize the total profit of selected items while ensuring the total weight does not exceed a given capacity.
# Implementation
The solution uses the Artificial Bee Colony algorithm to iteratively improve a population of binary solutions. The fitness of solutions is evaluated using a penalized objective function that considers both the profit and weight constraints.
# Requirements
Python 3.x
NumPy
Install dependencies using:
pip install numpy
# Usage
To run the project:
python knapsack_abc.py
# Examples
The program solves multiple instances of the knapsack problem, providing the best solution and its corresponding profit for each instance.
# Output
The output includes:
Best solution for each instance.
Maximum profit for each instance.
Statistical measures (mean, standard deviation, max, min) of the results across multiple runs
##References
