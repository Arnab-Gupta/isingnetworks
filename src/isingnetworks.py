import networkx as nx
import numpy as np
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

class IsingModel():
    
    def __init__(self, graph):
        
        self.name = "IsingModel"
        self.size = graph.number_of_nodes()
        self.graph = graph
        self.list_of_neigh = {}
        for node in self.graph.nodes():
            self.list_of_neigh[node] = list(self.graph.neighbors(node))
        
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
        for i in range(self.size):
            ss = np.sum(self.state[self.list_of_neigh[i]])
            en += self.state[i] * ss
        return -0.5 * self.J * en
    
    def __montecarlo(self):
        # pick a random source node
        beta = 1/self.temperature
        rsnode = np.random.randint(0, self.size)
        # get the spin of this node
        s = self.state[rsnode]
        # sum of all neighbouring spins
        ss = np.sum(self.state[self.list_of_neigh[rsnode]])
        # transition energy
        delE = 2.0 * self.J * ss * s
        # calculate transition probability
        prob = math.exp(-delE * beta)
        # conditionally accept the transition
        if prob > random.random():
            s = -s
        self.state[rsnode] = s
        
    def simulate(self, temperature, energy = True):
        
        self.temperature = temperature
    
        self.initialize(self.initial_state)
        # initialize spin vector
        
        if energy == True:
            for i in range(self.iterations):
                self.__montecarlo()
                mag = self.__netmag()
                ene = self.__netenergy()
            return np.abs(mag)/float(self.size), ene
        else:
            for i in range(self.iterations):
                self.__montecarlo()
                mag = self.__netmag()
            return np.abs(mag)/float(self.size)

    def simulate_decay(self, temperature):
        
        self.temperature = temperature
        
        self.initialize(self.initial_state)
        
        for i in range(self.iterations):
            self.__montecarlo()
            mag = self.__netmag()
            if mag <= (0.75*self.orig_mag):
                return i
        
    def sim_fast(self, temperature):
        return self.simulate(temperature, energy = False)
    
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
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.simulate)(i) for i in temperature)
    
        for cr in results:
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
    
    def viz_fast(self, temperature):
        """Simulate and visualise the magnetization wrt a temperature range.
        
        Parameters
        ----------
        temperature: array_like
            This is the temperature range over which the model shall be simulated.

        """
        mag = np.zeros(len(temperature))
        
        for i in tqdm(range(len(temperature))):
            # print(" Temp : " + str(temperature[i]))
            mag[i] = self.simulate(temperature[i], energy = False)
        
        plt.figure()
        plt.plot(temperature, mag)
        plt.xlabel('Temperature')
        plt.ylabel('Magnetization')
        plt.title('Magnetization vs Temperature')
        
        return mag
    
    def viz_fast_parallel(self, temperature):
        """Simulate and visualise the magnetization wrt a temperature range with python parallelization.
        
        Parameters
        ----------
        temperature: array_like
            This is the temperature range over which the model shall be simulated.

        """
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.sim_fast)(i) for i in temperature)
        
        print(results)
        
        plt.figure()
        plt.plot(temperature, results)
        plt.xlabel('Temperature')
        plt.ylabel('Magnetization')
        plt.title('Magnetization vs Temperature')
        
        return results

    def viz_decay(self, N, temperature, ensemble=5):
        """A simulation which returns number of steps required for the magnetization in the network to decay to 0.75 of original value.

        Parameters
        ----------
        N: int
            size of the network

        temperature: array_like
            temperature range over which the model is simulated

        ensemble: number of samples from which the median is considered

        Returns
        -------
        the decay time array

        """
        self.orig_mag = N
        tau = np.zeros(len(temperature))
        tau_2 = np.zeros(ensemble)
        for i in tqdm(range(len(temperature))):
            for j in range(ensemble):
                tau_2[j] = self.simulate_decay(temperature[i])
            tau[i] = np.median(tau_2)
            
        plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.plot(temperature, tau, 'x')
        plt.xlabel('Temperature')
        plt.ylabel('Tau')
        plt.xscale('log')
        plt.yscale('log')
        
        return tau

    def curie_temp(self, end, ensemble, start=0.1, threshold=0.1):
        """Determines the Curie temperature / critical temperature

        Parameters
        ----------

        end: float
            upper limit of the temperature range

        ensemble: int
            number of samples from which the median is evaluated

        start: float
            lower limit of the temperature range

        threshold: float
            value to which the original magnetization must decay at the Curie temperature
        """
        
        orig_mag = self.size
        
        step_size = (end - start)/30
        
        res = np.zeros(ensemble)
        
        temperature = np.arange(start, end, step_size)
        for t in tqdm(range(ensemble)):
            for i in tqdm(range(len(temperature))):
                # print(" Temp : " + str(temperature[i]))
                mag = self.simulate(temperature[i], energy = False)
                if mag < threshold:
                    res[t] = temperature[i]
                    break                    
                    
        return np.median(res)