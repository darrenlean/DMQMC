#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:35:05 2021

@author: darrenlean
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import Hamiltonians as H

def analyse_Hamiltonian(matrix):
    '''
    Statistical analysis and visualisation of the distributions of the 
    diagonals and off-diagonals of any given Hamiltonian matrix 
    '''
    
    startTime = time.time()
    
    # Separating the diagonals and the off-diagonals
    diagonals = np.copy(np.diagonal(matrix))
    np.fill_diagonal(matrix, 0)
    off_diagonals = np.ndarray.flatten(matrix)
    np.fill_diagonal(matrix, diagonals)
    
    # Filter out the zero matrix entries because they don't participate in DMQMC
    diagonals = diagonals[diagonals != 0]
    off_diagonals = off_diagonals[off_diagonals != 0]
    
    # Building statistics of both groups
    diag_unique, diag_counts = np.unique(diagonals, return_counts=True)
    diag_prob = diag_counts/sum(diag_counts)
    diag_average = np.average(diagonals)
    diag_std = np.std(diagonals)
    
    off_diag_unique, off_diag_counts = np.unique(off_diagonals, return_counts=True)
    off_diag_prob = off_diag_counts/sum(off_diag_counts)
    off_diag_average = np.average(off_diagonals)
    off_diag_std = np.std(off_diagonals)
        
    # Setting up the plot
    plt.figure(1)
    plt.grid()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel("Unique entry (neglecting zeros)")
    plt.ylabel("Probability")
    
    # Plotting diagonal entry statistics
    plt.plot(diag_unique, diag_prob, 'x', color = 'blue', label = 'Diagonal entries')
    plt.plot([diag_average, diag_average], [0, 1], color = 'blue', \
             label = "Diagonal average with one std")
    plt.plot([diag_average-diag_std, diag_average+diag_std], \
             [0.5*max(diag_prob), 0.5*max(diag_prob)], color = 'blue')

    # Plotting off-diagonal entry statistics
    plt.plot(off_diag_unique, off_diag_prob, 'x', color = 'red', label = 'Off-diagonal entries')
    plt.plot([off_diag_average, off_diag_average], [0, 1], color = 'red', \
             label = "Off-diagonal average with one std")
    plt.plot([off_diag_average-off_diag_std, off_diag_average+off_diag_std], \
             [0.5*max(off_diag_prob), 0.5*max(off_diag_prob)], color = 'red')
    
    plt.legend()
    
    #plt.figure(2)
    #plt.imshow(matrix, cmap='cool', interpolation='nearest')
    
    endTime = time.time()
    print('Time taken to run analysis', (endTime-startTime)/60, 'minutes')
    
    
# Parameters of the Heisenberg system
N = 10
J1 = 1
J2 = 1

startTime = time.time() 
# Constructing the matrix
matrix = H.Frustrated_Ladder(N, J1, J2).Hamiltonian_matrix()
# matrix = H.NN_Chain(N, J1).Hamiltonian_matrix()
endTime = time.time()
print('Time taken to generate the matrix', (endTime-startTime)/60, 'minutes')

analyse_Hamiltonian(matrix)