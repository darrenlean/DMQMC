#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class performing exact diagonalisation on some Hamiltonian
"""

 
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [10, 5.25]
   } 
plt.rcParams.update(params)

class ED:
    def __init__(self, N, beta, resolution, Hamiltonian):
        
        #Number of spins
        self.__N = N
        #Final beta of the simulation
        self.__beta = beta
        #Number of targeted betas between 0 and final beta
        self.__resolution = resolution
        #Hamiltonian matrix of the system (MUST BE A MATRIX)
        self.__hamiltonian = Hamiltonian

        #Energy eigenvalues
        self.__evalues = []
        #Unitary transformation matrix = columns of eigenvectors lined up
        #according to the energy eigenvalues
        self.__U = []
        
        #Storage for the results
        
        #Various betas between 0 and final beta
        self.__result_betas = []
        #analytic energies of corresponding betas
        self.__result_energies = []
        
        #Run exact diagonalisation of the Hamiltonian
        self.diagonalisation()
        #Compute the energies at various betas
        self.compute_all_energies()
    
    def diagonalisation(self):
        '''
        Performs exact diagonalisation of the hamiltonian and store the
        eigenvalues and corresponding eigenvectors
        '''
        #scipy exact diagonalisation
        evalues, evectors = la.eig(self.__hamiltonian, left = False, right = True)
        
        #Hamiltonian is real and symmetric so eigenvalues are all real
        self.__evalues = evalues.real
        #scipy gives the eigenvectors in rows so we transpose to get the column
        self.__U = np.transpose(evectors)
        
    def compute_energy(self, target_beta):
        '''
        Returns the energy at a given beta
        '''
        
        #rho_diag is the density matrix in the hamiltonian eigenbasis
        #In that basis, rho_diag is diagonal
        #this identity matrix is an initialiser
        rho_diag = np.identity(len(self.__hamiltonian))
        
        #rho_diag = diag(e^-beta*eigenvalues)
        for i in range(len(self.__hamiltonian)):
            rho_diag[i][i] = np.exp(-target_beta*self.__evalues[i])
        
        #Now transform rho into the system basis via
        #rho = U*rho_diag*U_transposed
        rho = np.matmul(np.transpose(self.__U), np.matmul(rho_diag, self.__U))
        
        #The energy by projection method is Tr(H*rho)/Tr(rho)
        E = np.trace(np.matmul(self.__hamiltonian, rho))/np.trace(rho)
        
        #returns the result energy
        return E
    
    def compute_all_energies(self):
        '''
        Computes and stores the energies for various betas
        '''
        
        #Clear memory for new results
        self.__result_betas = []
        self.__result_energies = []
        
        #Running compute_energy for the different betas
        for i in range(self.__resolution):
            target_beta = (i+1)*self.__beta/self.__resolution
            #print('Computing beta = ', target_beta)
            self.__result_betas.append(target_beta)
            self.__result_energies.append(self.compute_energy(target_beta))
            
    def get_result(self):
        '''
        Returns list of betas and their corresponding energies
        '''
        return self.__result_betas, self.__result_energies
        
    def show_result(self):
        '''
        Plots graph of energy and various beta
        '''
        print('Hamiltonian matrix')
        print(self.__hamiltonian)
        print('Max eigenvalue')
        print(self.__evalues.max())
        print('Min eigenvalue')
        print(self.__evalues.min())
        #plt.title(self.__filename+' Exact Diagonalisation')
        plt.plot(self.__result_betas, self.__result_energies, label='Exact Diagonalisation')
        plt.legend()
        
    def show_hamiltonian(self):
        '''
        Print the hamiltonian of the system
        '''
        print(self.__hamiltonian)


