#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:35:05 2021

@author: darrenlean
"""

import numpy as np
import matplotlib.pyplot as plt
import Hamiltonians as H

# Parameters of the Heisenberg system
N = 8
J1 = 1
J2 = 1

# Constructing the matrix
matrix = H.Frustrated_Ladder(N, J1, J2).Hamiltonian_matrix()

def analyse_Hamiltonian(matrix):
    
    # Separating the diagonals and the off-diagonals
    diagonals = np.copy(np.diagonal(matrix))
    np.fill_diagonal(matrix, 0)
    off_diagonals = np.ndarray.flatten(matrix)
    
    # Filter out the zero matrix entries because they don't participate in DMQMC
    diagonals = diagonals[diagonals != 0]
    off_diagonals = off_diagonals[off_diagonals != 0]
    
    # Building statistics of both groups
    diag_unique, diag_counts = np.unique(diagonals, return_counts=True)
    diag_average = np.average(diagonals)
    diag_std = np.std(diagonals)
    
    off_diag_unique, off_diag_counts = np.unique(off_diagonals, return_counts=True)
    off_diag_average = np.average(off_diagonals)
    off_diag_std = np.std(off_diagonals)
        
    # Plot histograms of both groups
    plt.figure(1)
    plt.title("Diagonal entries (neglecting zeros)")
    plt.grid()
    plt.xlabel("Unique entry")
    plt.ylabel("Count")
    plt.plot(diag_unique, diag_counts, 'x')
    plt.vlines(diag_average, 0, max(diag_counts), label = "average")
    plt.plot([diag_average-diag_std, diag_average+diag_std], \
             [0.5*max(diag_counts), 0.5*max(diag_counts)], label = "one std")
    plt.legend()
    
    plt.figure(2)
    plt.title("Off-diagonal entries (neglecting zeros)")
    plt.grid()
    plt.xlabel("Unique entry")
    plt.ylabel("Count")
    plt.plot(off_diag_unique, off_diag_counts, 'x')
    plt.vlines(off_diag_average, 0, max(off_diag_counts), label = "average")
    plt.plot([off_diag_average-off_diag_std, off_diag_average+off_diag_std], \
             [0.5*max(off_diag_counts), 0.5*max(off_diag_counts)], label = "one std")
    plt.legend()
    
analyse_Hamiltonian(matrix)