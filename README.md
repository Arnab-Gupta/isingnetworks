# Ising Model on Networks

This is a python package which simulates the Ising model on complex networks.

Given any network built using the networkx package as input, this package runs Monte Carlo simulations on random source nodes and evaluates the model according to the Metropolis algorithm, and returns the magnetization and the energy of the system.

Version 1.0.2: This has an added visualizer to plot the output graphs.

Version 1.0.4: The Github repo has an added ipynb that can be referred to for instructions about usage.

## Installation

Run the following to install:
```python
pip install isingnetworks
```

## Usage 

Import the class:
```python
from isingnetworks import IsingModel
```

Create a graph (g) using networkx.

Create an intance of the IsingModel class:
```python
model = IsingModel(g)
```

Using the viz function:
```python
model.viz(J, temperature_array, no_of_iterations, initial_state)
```