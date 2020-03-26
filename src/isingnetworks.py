import networkx as nx
import numpy as np
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class IsingModel():
    
    def __init__(self, graph):
        
        self.name = "IsingModel"
        self.size = graph.number_of_nodes()
        self.graph = graph
        
    def initialize(self, initial_state):
        
        self.state = np.random.choice([-1,1], self.size, p=[1-initial_state, initial_state])
    
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
        
    def simulate(self, J, temperature, iterations, initial_state, tqdmx = True):
        
        if np.abs(initial_state) > 1:
            raise Exception("initial_state should be between 0 and 1")
        
        self.J = J
        self.temperature = temperature
        
        adj_matrix = nx.adjacency_matrix(self.graph)
        top = adj_matrix.todense()
    
        self.initialize(initial_state)
        # initialize spin vector
        if tqdmx == True:
            for i in tqdm(range(iterations)):
                self.__montecarlo(top)
                mag = self.__netmag()
                ene = self.__netenergy()
        else:
            for i in (range(iterations)):
                self.__montecarlo(top)
                mag = self.__netmag()
                ene = self.__netenergy()
            
        return np.abs(mag)/float(self.size), ene
    
    def viz(self, J, temperature, iterations, initial_state):
        
        mag = np.zeros(len(temperature))
        ene = np.zeros(len(temperature))
        
        for i in tqdm(range(len(temperature))):
            print(" Temp : " + str(temperature[i]))
            mag[i], ene[i] = self.simulate(J, temperature[i], iterations, initial_state, tqdmx = False)
        
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
        
        
        
        


