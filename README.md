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
#References
Step by Step Procedure of ABC (1)
Burak Kaya, İbrahim Eke. DEVELOPMENTS IN ARTIFICIAL BEE COLONY ALGORITHM AND THE RESULTS (2)
Zeynep Banu Özger, Bülent Bolat, Banu Diri. (2020). A Locally Searched Binary Artificial Bee Colony Algorithm Based on Hamming Distance for Binary Optimization. Journal of Natural and Applied Sciences, 24(1), 120-131. (3)
Akay, B., & Karaboğa, D. (2009). Performance Analysis of the Artificial Bee Colony (ABC) Algorithm in Numerical Optimization Problems. Supported by Erciyes University Scientific Research Projects Unit, Project Code: FBA-06-22. (4)
