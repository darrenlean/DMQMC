#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:55:02 2021

@author: darrenlean
"""

"""

This script contains all the classes for the Hamiltonians

Parent class 1: Hamiltonian()
Attributes: size of the Hilbert space
Methods: Hamiltonian_matrix() returns the Hamiltonian in matrix form
         symmetric_check() returns boolean on whether the Hamiltonian matrix is
                           symmetric
         
Parent class 2: Heisenberg_Model()
All Heisenberg Models are based on neighbour (of certain order) interactions
This class inherits the Hamiltonian class
Attributes: Number of spins N, Coupling strength J
Methods: connected_sites(site) returns all connected sites of a given site
         Hamiltonian(row_index, col_index) returns entry of Hamiltonian at a 
                                             set of given matrix indices
                                             
All the children of Heisenberg_Model() require the 
neighbouring(single_spin_site) method and are sufficient with it.
The neighbours(single_spin_site) method is specific to the topology of the 
Heisenberg model of interest. It should return the (list of) neighbours of 
a given single-spin site (index)

Naming convention for Heisenberg_Model children: OrderOfInteraction_Topology()

Order of Interaction:
NN = Nearest Neighbour (Interaction)
NNN = Next Nearest Neighbour (Interaction)

Topology: 
Chain = 1D chain 
Ladder = (2*integer) chain 

Currently, there are
NN_Chain(N, J)
NNN_Chain(N, J)
NN_Ladder(N, J)
NNN_Ladder(N, J)
***Triangular***

"""

import numpy as np

class Hamiltonian():
    
    def __init__(self, Hilbert_size):
        # Hilbert_size is the dimension of the Hilbert space
        # The Hamiltonian is a dim(Hilbert space) * dim(Hilbert_space) 
        # and symmetric matrix
        self.__Hilbert_size = Hilbert_size
        
    def Hamiltonian_matrix(self):
        '''
        Returns the Hamiltonian matrix given a Hamiltonian method that returns
        the entries of the matrix
        '''
        
        # The Hamiltonian is a dim(Hilbert space) * dim(Hilbert_space) matrix
        H = np.zeros([self.__Hilbert_size, self.__Hilbert_size])
    
        #Running for all entries
        for i in range(self.__Hilbert_size):
            for j in range(self.__Hilbert_size):
                H[i][j] = self.Hamiltonian(i, j)
                
        #Return the matrix
        return H
    
    def symmetric_check(self):
        '''
        Checks if the Hamiltonian matrix is symmetrical by comparing it with 
        its transpose
        '''
        # Obtaining the Hamiltonian matrix
        H = self.Hamiltonian_matrix()
        #Checking if every corresponding entries in the original and transposed
        # matrix are the same
        if H.all() == np.transpose(H).all():
            return True
        else:
            return False
    
class Heisenberg_Model(Hamiltonian):
    
    def __init__(self, N, J=1):
        # Number of spins
        self.__N = N
        # Coupling strength
        self.__J = J
        # Inheritance of Hamiltonian_matrix and symmetric_check methods
        Hamiltonian.__init__(self, 2**N)
    
    def connected_sites(self, site):
        '''
        Returns the decimal system connected states of decimal system site
        in a N-spin Heisenberg model
        site is a state of up and down spins when considering it in a binary 
        system
        '''
        #storage for connected states
        connected_indices = []
        #decimal to binary; [2:] to remove the '0b'
        state = bin(site)[2:]
    
        #Python bin gives the binary number up to its leading one
        #Sometimes this misses out the zeros in front of the leading one
        #We add the front zero if the number of states is less than the specified N
        state = (self.__N-len(state))*'0' + state
        
        #looping through all the single-spin states
        for j in range(0, len(state)):
            
            #k are the nearest single-spin neighbour of single-spin j
            k = self.neighbours(j)
            
            #looping through all the nearest neighbours
            for a in range(len(k)):
                #checking for antiparallel for each of the k neighbours
                #If antiparallel
                if state[j] != state[int(k[a])]:
                    temp = list(state)
                    #Flipping the antiparallel pair to obtain the connected 
                    # state
                    temp[j] = str(abs(int(state[j])-1))
                    temp[int(k[a])] = str(abs(int(state[int(k[a])])-1))
                    connected_state = "".join(temp)
                    #binary to decimal, e.g. int('1010', 2) = 10
                    connected_indices.append(int(connected_state, 2))
        
        #Remove repeated indices
        connected_indices = list(dict.fromkeys(connected_indices))
        
        #Return the results
        return connected_indices
    
    def Hamiltonian(self, row_index, col_index):
        '''
        Generate entry of the Hamiltonian matrix in a Heisenberg model given
        a state and its connected states
        '''
        
        # row_index is the state in decimal system
        connected_states = self.connected_sites(row_index)
        
        # Number of antiparallel pairs = Number of connected states
        Na = len(connected_states)
        # Number of parallel pairs + Number of antiparallel pairs = N - 1
        Np = self.__N - 1 - Na
        
        # Diagonal entry
        if row_index == col_index:
            return self.__J*(Np - Na)/4 
        # Off-diagonal connected entry
        elif col_index in connected_states:
            return 0.5*self.__J
        # Off-diagonal unconnected entry
        else:
            return 0

class NN_Chain(Heisenberg_Model):
    
    def __init__(self, N, J):
        '''
        Nearest Neighbour Interaction Heisenberg model 1D Closed Chain
        '''
        if N < 2:
            raise ValueError("N must be a positive integer of at least 2!")
        # Number of spins
        self.__N = int(N)
        # Coupling strength
        self.__J = J
        # Inheritance of Inheritance of Hamiltonian_matrix, symmetric_check,
        # connected_sites and Hamiltonian methods
        Heisenberg_Model.__init__(self, self.__N, self.__J)
        
    def neighbours(self, single_spin_site):
        '''
        Returns the neighbouring single spin indices of a single spin site
        '''
        j = single_spin_site
        #the nearest neighbours of j are k = j +/- 1
        if self.__N < 3:
            #open chain condition for two-spin system
            if j == 0:
                k = [j+1]
            elif j == self.__N-1:
                k = [j-1]
            else:
                k = [j-1, j+1]
        else:
            #closed chain conditions
            if j == 0:
                k = [self.__N-1, j+1]
            elif j == self.__N-1:
                k = [j-1, 0]
            else:
                k = [j-1, j+1]
                
        #Return the neighbours
        return k
    
class NNN_Chain(Heisenberg_Model):
    
    def __init__(self, N, J):
        '''
        Next Nearest Neighbour Interaction Heisenberg model 1D Closed Chain
        '''
        if N < 3:
            raise ValueError("N must be a positive integer of at least 3!")
        # Number of spins
        self.__N = int(N)
        # Coupling strength
        self.__J = J
        # Inheritance of Inheritance of Hamiltonian_matrix, symmetric_check,
        # connected_sites and Hamiltonian methods
        Heisenberg_Model.__init__(self, self.__N, self.__J)
        
    def neighbours(self, single_spin_site):
        '''
        Returns the neighbouring single spin indices of a single spin site
        '''
        j = single_spin_site
        #the nearest neighbours of j are k = j +/- 2
        if self.__N < 4:
            #open chain condition for two-spin system
            if j == 0:
                k = [j+2]
            elif j == 1:
                k = []
            else:
                k = [j-2]
        else:
            #closed chain conditions
            if j == 0:
                k = [self.__N-2, j+2]
            elif j == 1:
                k = [self.__N-1, j+2]
            elif j == self.__N-1:
                k = [j-2, 1]
            elif j == self.__N-2:
                k = [j-2, 0]
            else:
                k = [j-2, j+2]
                
        #Return the neighbours
        return k
    
class NN_Ladder(Heisenberg_Model):
    
    def __init__(self, N, J):
        '''
        Nearest Neighbour Interaction Heisenberg model Ladder lattice
        '''
        
        if N%2 != 0 or N < 4:
            raise ValueError("N must be an even positive integer of at least 4!")
        # Number of spins
        self.__N = int(N)
        # Coupling strength
        self.__J = J
        # Inheritance of Inheritance of Hamiltonian_matrix, symmetric_check,
        # connected_sites and Hamiltonian methods
        Heisenberg_Model.__init__(self, self.__N, self.__J)
        
    def neighbours(self, single_spin_site):
        '''
        Returns the neighbouring single spin indices of a single spin site
        '''
        j = single_spin_site
        #change of labeling j -> j' = j-N for j=0->N-1 
        #and j' = j-N+1 for j=N->2N-1
        if j <= self.__N/2-1:
            j -= self.__N/2
        else:
            j -= self.__N/2 - 1
        
        #the nearest neighbours of j' are k' = -j', j'+/- 1
        if self.__N == 4:
            #open chain conditions
            if j == 1:
                k = [-j, j+1]
            elif j == -1:
                k = [-j, j-1]
            elif j == self.__N/2:
                k = [-j, j-1]
            elif j == -self.__N/2:
                k = [-j, j+1]
            else:
                k = [-j, j-1, j+1]
        else:
            #close chain conditions
            if j == 1:
                k = [-j, self.__N/2, j+1]
            elif j == -1:
                k = [-j, j-1, -self.__N/2]
            elif j == self.__N/2:
                k = [-j, j-1, 1]
            elif j == -self.__N/2:
                k = [-j, -1, j+1]
            else:
                k = [-j, j-1, j+1]
        
        #change of labeling k' -> k = k'+N for k' = -N->-1
        #and k = k'+N-1 for k' = 1->N
        for a in range(len(k)):
            if k[a] <= -1:
                k[a] = int(k[a] + self.__N/2)
            else:
                k[a] = int(k[a] + self.__N/2 - 1)
        
        #Return the neighbours
        return k

class NNN_Ladder(Heisenberg_Model):
    
    def __init__(self, N, J):
        '''
        Next Nearest Neighbour Interaction Heisenberg model Ladder lattice
        '''
        
        if N%2 != 0 or N < 6:
            raise ValueError("N must be an even positive integer of at least 6!")
        # Number of spins
        self.__N = int(N)
        # Coupling strength
        self.__J = J
        # Inheritance of Inheritance of Hamiltonian_matrix, symmetric_check,
        # connected_sites and Hamiltonian methods
        Heisenberg_Model.__init__(self, self.__N, self.__J)
        
    def neighbours(self, single_spin_site):
        '''
        Returns the neighbouring single spin indices of a single spin site
        '''
        j = single_spin_site
        #change of labeling j -> j' = j-N for j=0->N-1 
        #and j' = j-N+1 for j=N->2N-1
        if j <= self.__N/2-1:
            j -= self.__N/2
        else:
            j -= self.__N/2 - 1
        
        #the nearest neighbours of j' are k' = -j' +/- 1
        if self.__N == 6:
            #open chain conditions
            if j == 1:
                k = [-2]
            elif j == -1:
                k = [2]
            elif j == self.__N/2:
                k = [-j+1]
            elif j == -self.__N/2:
                k = [-j-1]
            else:
                k = [-j-1, -j+1]
        else:
            #close chain conditions
            if j == 1:
                k = [-2, -self.__N/2]
            elif j == -1:
                k = [self.__N/2, 2]
            elif j == self.__N/2:
                k = [-1, -j+1]
            elif j == -self.__N/2:
                k = [-j-1, 1]
            else:
                k = [-j-1, -j+1]
        
        #change of labeling k' -> k = k'+N for k' = -N->-1
        #and k = k'+N-1 for k' = 1->N
        for a in range(len(k)):
            if k[a] <= -1:
                k[a] = int(k[a] + self.__N/2)
            else:
                k[a] = int(k[a] + self.__N/2 - 1)
        
        #Return the neighbours
        return k

class Frustrated_Ladder(Hamiltonian):
    
    def __init__(self, N, J1, J2):
        '''
        Heisenberg model Frustrated Ladder lattice
        '''
        
        if N%2 != 0 or N < 6:
            raise ValueError("N must be an even positive integer of at least 6!")
        # Number of spins
        self.__N = int(N)
        # Coupling strength
        self.__J1 = J1
        self.__J2 = J2
        
        # Inheritance of Hamiltonian_matrix and symmetric_check methods
        Hamiltonian.__init__(self, 2**N)
        
        #Initialising NN and NNN
        self.__NN = NN_Ladder(self.__N, self.__J1)
        self.__NNN = NNN_Ladder(self.__N, self.__J2)
        
    def connected_sites(self, single_spin_site):
        '''
        Returns the decimal system connected states of decimal system site
        in a N-spin Heisenberg model
        site is a state of up and down spins when considering it in a binary 
        system
        '''
        # Connected sites from nearest neighbour interaction
        connected_indices = self.__NN.connected_sites(single_spin_site)
        # Connected sites from nearest neighbour interaction
        connected_indices += self.__NNN.connected_sites(single_spin_site)
        # Remove repeated indices
        connected_indices = list(dict.fromkeys(connected_indices))
        # Return the results
        return connected_indices
    
    def Hamiltonian(self, row_index, col_index):
        '''
        Generate entry of the Hamiltonian matrix in a Heisenberg model given
        a state and its connected states
        '''
        
        return self.__NN.Hamiltonian(row_index, col_index) + \
                    self.__NNN.Hamiltonian(row_index, col_index)
        

class NN_Triangular(Heisenberg_Model):
    
    def __init__(self, N, J):
        '''
        Neighbour Interaction Heisenberg model 4-spin Triangular Lattice
        '''
        
        # Set the limitations of the number of spins
        if N != 4:
            raise ValueError("N must be 4!")
        # Number of spins
        self.__N = int(N)
        # Coupling strength
        self.__J = J
        # Inheritance of Inheritance of Hamiltonian_matrix, symmetric_check,
        # connected_sites and Hamiltonian methods
        Heisenberg_Model.__init__(self, self.__N, self.__J)
        
    
    def neighbours(self, single_spin_site):
        '''
        Returns the neighbouring single spin indices of a single spin site
        '''
        j = single_spin_site
        #change of labeling j -> j' = j-N for j=0->N-1 
        #and j' = j-N+1 for j=N->2N-1
        if j <= self.__N/2-1:
            j -= self.__N/2
        else:
            j -= self.__N/2 - 1
            
        #the nearest neighbours of j' are k' = -j', j'+/- 1 and 
        #(-j'-1 for j' < 0) or (-j'+1 for j' > 0)

        if j == 0:
            k = [1, 3]
        elif j == 1:
            k = [0, 2, 3]
        elif j == 2:
            k = [0, 1, 3]
        elif j == 3:
            k = [1, 2]
        else:
            k = []
            
        #Return the neighbours
        return k
   
class My_Template_New_Heisenberg(Heisenberg_Model):
    
    def __init__(self, N, J):
        '''
        *This is a tutorial of setting up a new Heisenberg model*
        *State the type of system here, for example:*
        Nearest Neighbour Interaction Heisenberg model 1D Closed Chain
        '''
        
        # Set the limitations of the number of spins
        if N < 2:
            raise ValueError("N must be a positive integer of at least 2!")
        # Number of spins
        self.__N = int(N)
        # Coupling strength
        self.__J = J
        # Inheritance of Inheritance of Hamiltonian_matrix, symmetric_check,
        # connected_sites and Hamiltonian methods
        Heisenberg_Model.__init__(self, self.__N, self.__J)
        
    
    def neighbours(self, single_spin_site):
        '''
        *Specify the topology of the system through this method*
        Returns the neighbouring single spin indices of a single spin site
        '''
        j = single_spin_site
        #the nearest neighbours of j are k = j +/- 1
        if self.__N < 3:
            #open chain condition for two-spin system
            if j == 0:
                k = [j+1]
            elif j == self.__N-1:
                k = [j-1]
            else:
                k = [j-1, j+1]
        else:
            #closed chain conditions
            if j == 0:
                k = [self.__N-1, j+1]
            elif j == self.__N-1:
                k = [j-1, 0]
            else:
                k = [j-1, j+1]
                
        #Return the neighbours
        return k

class Heisenberg_Triangular:
    
    def __init__(self, N, J):
        
        self.__N = N
        self.__J = J
        self.topology = []
        # Topology stores the PARTICLE indices that have bonds between them
        # similarly to an adjacency matrix in graph theory
        if (self.__N == 4):
            self.topology = [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]]
        else:
            return "Insufficient topology provided"
        # Creating topology with more particles is desired
        
        
    def coupling(self, i, j, S):
        '''
        Calculate coupling of 2 spins in state S to generate connected state
        if S has non-zero overlap if ith and jth spins
        '''        
        state = bin(S)[2:]
        conn_state = state
        # Correcting for the decimal
        state =  (self.__N - len((state)))*'0' + state
        
        # Only opposite spins can generate connected state
        if (state[i] == state[j]):
            return -1
        else:
            # Perform spin pair flip
            conn_state = state[:i] + str(abs(int(state[i])-1)) + state[i+1:j] + \
                                        str(abs(int(state[j])-1)) + state[j+1:]
            return int(conn_state, 2)
        
    def connected_sites(self, site):
        
        conn_sites = []
        for bond in self.topology:
            # Calculate coupled state for all possible bonds
            coupled_state = self.coupling(bond[0], bond[1], site)
            if (coupled_state != -1):
                conn_sites.append(coupled_state)
                
        return conn_sites
    
    def Hamiltonian(self, Sa, Sb):
        
        Sa_conn_sites = self.connected_sites(Sa)
        
        anti_parallel_couplings = len(Sa_conn_sites)
        parallel_couplings = len(self.topology) - anti_parallel_couplings
        
        if Sb not in Sa_conn_sites:
            return 0
        if (Sa == Sb):
            return (self.__J/4)*(parallel_couplings - anti_parallel_couplings)
        else:
            return self.__J/2
    
    def Hamiltonian_matrix(self):
        matrix_H = np.zeros([2**self.__N, 2**self.__N])
        
        for i in range(0, 2**self.__N):
            for j in range(0, 2**self.__N):
                matrix_H[i][j] = self.Hamiltonian(i, j)
        for i in range(0, 2**self.__N):
            for j in range(0, 2**self.__N):
                if matrix_H[i][j] != matrix_H[j][i]:
                    print('ASSIMETRY FOUND')
                    print(i, j)
                    
                
        return matrix_H

