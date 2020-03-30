import networkx as nx
import numpy as np
import math
import random
from tqdm import tqdm
import concurrent.futures as cf
import matplotlib.pyplot as plt

class IsingModel():
    
    def __init__(self, graph):
        
        self.name = "IsingModel"
        self.size = graph.number_of_nodes()
        self.graph = graph
        
    def initialize(self, initial_state):
        
        self.state = np.random.choice([-1,1], self.size, p=[1-initial_state, initial_state])

    def set_J(self, J):
        """Set the value of J

        Parameter(s)
        ----------
        J : int
            This is the interaction coefficient.
        """
        self.J = J

    def set_iterations(self, iterations):
        """Set your desired number of iterations per temperature value

        Parameter(s)
        ----------
        iterations: int
            This is the number of iterations per temperature value.
        """
        self.iterations = iterations

    def set_initial_state(self, initial_state):
        """Set initial state

        Parameter(s):
        initial_state: int [0,1]
            This is the initial state of all nodes of the system.
        """
        if np.abs(initial_state) > 1:
            raise Exception("initial_state should be between 0 and 1")
        
        self.initial_state = initial_state
    
    def __netmag(self):
        
        return np.sum(self.state)
    
    def __netenergy(self):
        en = 0.
        adj_matrix = nx.adjacency_matrix(self.graph)
        top = adj_matrix.todense()
        for i in range(self.size):
            ss = np.sum(self.state[top[i].nonzero()[1]])
            en += self.state[i] * ss
        return -0.5 * self.J * en
    
    def __montecarlo(self, top):
        # pick a random source node
        beta = 1/self.temperature
        rsnode = np.random.randint(0, self.size)
        # get the spin of this node
        s = self.state[rsnode]
        # sum of all neighbouring spins
        ss = np.sum(self.state[top[rsnode].nonzero()[1]])
        # transition energy
        delE = 2.0 * self.J * ss * s
        # calculate transition probability
        prob = math.exp(-delE * beta)
        # conditionally accept the transition
        if prob > random.random():
            s = -s
        self.state[rsnode] = s
        
    def simulate(self, temperature):
        
        self.temperature = temperature
        
        adj_matrix = nx.adjacency_matrix(self.graph)
        top = adj_matrix.todense()
    
        self.initialize(self.initial_state)
        # initialize spin vector
        
        for i in range(self.iterations):
            self.__montecarlo(top)
            mag = self.__netmag()
            ene = self.__netenergy()
            
        return np.abs(mag)/float(self.size), ene

    
    def viz(self, temperature):
        """Simulate and visualise the energy and magnetization wrt a temperature range.
        
        Parameters
        ----------
        temperature: array_like
            This is the temperature range over which the model shall be simulated.

        """
        mag = np.zeros(len(temperature))
        ene = np.zeros(len(temperature))
        
        for i in tqdm(range(len(temperature))):
            # print(" Temp : " + str(temperature[i]))
            mag[i], ene[i] = self.simulate(temperature[i])
        
        plt.figure()
        plt.plot(temperature, mag)
        plt.xlabel('Temperature')
        plt.ylabel('Magnetization')
        plt.title('Magnetization vs Temperature')
        
        plt.figure()
        plt.plot(temperature, ene)
        plt.xlabel('Temperature')
        plt.ylabel('Energy')
        plt.title('Energy vs Temperature')
        
        return mag, ene

    def viz_parallel(self, temperature):
        """Simulate and visualise the energy and magnetization wrt a temperature range with python parallelization.
        
        Parameters
        ----------
        temperature: array_like
            This is the temperature range over which the model shall be simulated.

        """

        mag = []
        ene = []
        with cf.ProcessPoolExecutor() as ex:
            results = ex.map(self.simulate, [i for i in temperature])

        comb_res = []
        for r in results:
            comb_res.append(r)

        # store the values of magnetization and energy returned by ex.map() into their respective arrays
        for cr in comb_res:
            mag.append(cr[0])
            ene.append(cr[1])

        plt.figure()
        plt.plot(temperature, mag)
        plt.xlabel('Temperature')
        plt.ylabel('Magnetization')
        plt.title('Magnetization vs Temperature')

        plt.figure()
        plt.plot(temperature, ene)
        plt.xlabel('Temperature')
        plt.ylabel('Energy')
        plt.title('Energy vs Temperature')

        return mag, ene
