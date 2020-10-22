# Ising Model on Networks

This is a python package which simulates the Ising model on complex networks.

Given any network built using the networkx package as input, this package runs Monte Carlo simulations on random source nodes and evaluates the model according to the Metropolis algorithm, and returns the magnetization and the energy of the system.

Version 1.0.2: This has an added visualizer to plot the output graphs.

Version 1.0.4: The Github repo has an added ipynb that can be referred to for instructions about usage.

Version 1.0.10: A new parallel implementation has been added.

Version 1.1.1: Two much faster functions have been added for both serial and parallel simulations.

Version 1.1.3: Added functions to calculate Curie temperature and decay time.

Version 1.1.4: Added a function to compute average degree

## Installation

Run the following to install:
```python
pip install isingnetworks
```

Additional packages required:
```python
pip install joblib
```


## Usage 

Import the class:
```python
from isingnetworks import IsingModel
```

Create a graph (g) using networkx.

Create an instance of the IsingModel class:
```python
model = IsingModel(g)
```

Using the viz function:
```python
model.set_J(val_of_J)
model.set_iterations(no_of_iterations)
model.set_initial_state(0) # 0 or 1

# Serial Implementation
model.viz(temperature_array)

# Parallel Implementation
model.viz_parallel(temperature_array)

# Faster Implementations
model.viz_fast(temperature_array)
model.viz_fast_parallel(temperature_array)
```
