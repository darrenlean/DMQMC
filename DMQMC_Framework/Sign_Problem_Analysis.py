#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sign problem analysis tool based on

Foulkes, Blunt, Spencer: The fermion sign problem in FCIQMC JChemPhys (2012)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:37:11 2021

@author: darrenlean
"""

import scipy as sp
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import Hamiltonians as H_generator
from numpy import histogram

params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [10, 5.25]
   } 
plt.rcParams.update(params)

def TmaxVmax(H):
    '''
    Returns the most positive eigenvalues of the T and V matrices given a
    hamiltonian H
    T matrix = - H matrix
    diag(T plus matrix) = diag(T matrix)
    off diag(T plus matrix) = positive off diag(T matrix) else zero
    diag(T minus matrix) = 0
    off diag(T minus matrix) = |negative off diag(T matrix)| else zero
    T matrix = T plus matrix - T minus matrix
    V matrix = T plus matrix + T minus matrix
    '''
    T = -H
    
    #Allocated memory for the T plus minus matrices
    Tplus = np.zeros([len(H), len(H)])
    Tminus = np.zeros([len(H), len(H)])
    for i in range(len(H)):
        for j in range(len(H)):
            #On the diagonal
            #All the diagonal entries go into T plus diagonal directly
            if i==j:
                Tplus[i][j] = T[i][j]
            #Off diagonal
            else:
                #If positive, goes to T plus
                if T[i][j] > 0:
                    Tplus[i][j] = T[i][j]
                #If negative, the absolute goes to T minus
                else:
                    Tminus[i][j] = abs(T[i][j])
                    
    #scipy exact diagonalisation for T = T plus - T minus
    print(Tplus)
    print(Tminus)
    Tevalues, Tevectors = la.eig(Tplus-Tminus, left = False, right = True)
    #the result required is the most positive eigenvalues of T
    #assuming that H is real symmetric -> T must be real symmetric
    # -> the eigenvalue must be purely real
    Tmax = max(Tevalues.real)

    #scipy exact diagonalisation for V = T plus + T minus
    Vevalues, Vevectors = la.eig(Tplus+Tminus, left = False, right = True)
    #the result required is the most positive eigenvalues of V
    #assuming that H is real symmetric -> V must be real symmetric
    # -> the eigenvalue must be purely real
    Vmax = max(Vevalues.real)
    
    print('Eigenvalues calculated')
    plt.plot(np.sort(Vevalues), label='v')
    print(Vevalues)
    plt.plot(np.sort(Tevalues), label='t')
    plt.legend()
    print(Tevalues)
    
    
    #returns the results
    return Tmax, Vmax

def DOS(H):

    T = -H
    
    #Allocated memory for the T plus minus matrices
    Tplus = np.zeros([len(H), len(H)])
    Tminus = np.zeros([len(H), len(H)])
    for i in range(len(H)):
        for j in range(len(H)):
            #On the diagonal
            #All the diagonal entries go into T plus diagonal directly
            if i==j:
                Tplus[i][j] = T[i][j]
            #Off diagonal
            else:
                #If positive, goes to T plus
                if T[i][j] > 0:
                    Tplus[i][j] = T[i][j]
                #If negative, the absolute goes to T minus
                else:
                    Tminus[i][j] = abs(T[i][j])
                    
    #scipy exact diagonalisation for T = T plus - T minus
    print(Tplus)
    print(Tminus)
    Tevalues, Tevectors = la.eig(Tplus-Tminus, left = False, right = True)
    #the result required is the most positive eigenvalues of T
    #assuming that H is real symmetric -> T must be real symmetric
    # -> the eigenvalue must be purely real
    t_evalues = np.sort(Tevalues.real)

    #scipy exact diagonalisation for V = T plus + T minus
    Vevalues, Vevectors = la.eig(Tplus+Tminus, left = False, right = True)
    #the result required is the most positive eigenvalues of V
    #assuming that H is real symmetric -> V must be real symmetric
    # -> the eigenvalue must be purely real
    v_evalues = np.sort(Vevalues.real)
    
    energy_bin = 0.25
    nT, binsT, patchesT = plt.hist(x=t_evalues, bins=15, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        
    #nV, binsV, patchesV = plt.hist(x=v_evalues, bins='auto', color='#0504aa',
                            #alpha=0.7, rwidth=0.85)
    
    plt.grid(axis='y', alpha=0.75)
    
    plt.xlabel('Eigenvalues')
    plt.ylabel('Density of states [1/E]')

class SP_Analysis:
    
    def __init__(self, N, J1, J2):
        
        self.__N = N
        self.__J1 = J1
        self.__J2 = J2
        
        self.__H = H_generator.Frustrated_Ladder(N, J1, J2).Hamiltonian_matrix()
        
        self.__T_evalues = []
        self.__V_evalues = []
        self.__operator = []
        
        # Unitary transformation matrices generating the diagonalisation of
        # T, V respectively 
        self.__U_T = []
        self.__U_V = []
        
        self.__combined = []
        
    def diagonalise(self):
        
        T = -self.__H
        #Allocated memory for the T plus minus matrices
        Tplus = np.zeros([2**(self.__N), 2**(self.__N)])
        Tminus = np.zeros([2**(self.__N), 2**(self.__N)])
        for i in range(2**(self.__N)):
            for j in range(2**(self.__N)):
                #On the diagonal
                #All the diagonal entries go into T plus diagonal directly
                if i==j:
                    Tplus[i][j] = T[i][j]
                #Off diagonal
                else:
                    #If positive, goes to T plus
                    if T[i][j] > 0:
                        Tplus[i][j] = T[i][j]
                    #If negative, the absolute goes to T minus
                    else:
                        Tminus[i][j] = abs(T[i][j])
                        
        #scipy exact diagonalisation for T = T plus - T minus
        print(Tplus)
        print(Tminus)
        Tevalues, Tevectors = la.eig(Tplus-Tminus, left = False, right = True)
        #the result required is the most positive eigenvalues of T
        #assuming that H is real symmetric -> T must be real symmetric
        # -> the eigenvalue must be purely real
        
        self.__T_evalues = Tevalues.real
        self.__U_T = np.transpose(Tevectors)
        
        #scipy exact diagonalisation for V = T plus + T minus
        Vevalues, Vevectors = la.eig(Tplus+Tminus, left = False, right = True)
        #the result required is the most positive eigenvalues of V
        #assuming that H is real symmetric -> V must be real symmetric
        # -> the eigenvalue must be purely real
        self.__V_evalues = Vevalues.real
        self.__U_V = np.transpose(Tevectors)
        
        self.__T_evalues = np.array(self.__T_evalues)
        self.__V_evalues = np.array(self.__V_evalues)
        self.__operator = np.array(2*Tminus)
        
        return self.__T_evalues[-1], self.__V_evalues[-1]
    
    def partition(self, beta, evalues):
        summa = 0
        for i in range(len(evalues)):
            summa += np.exp(evalues[i]*beta)
        
        return summa
    
    def free_energy(self, beta, evalues):
        return -1/beta*np.log(self.partition(beta, evalues))
    
    def expectation(self, beta, operator, evalues, U):
        #Transform operator into proper eigenbasis:
        
        operator_eigen = np.matmul(np.transpose(U), np.matmul(operator, U))

        trace = 0
        for i in range(len(evalues)):
            trace += operator_eigen[i][i]*np.exp(beta*evalues[i])
            
        return trace/self.partition(beta, evalues)
    
    def bogoliubov_analysis(self, final_beta, resolution):
        
        beta_range = np.linspace(0.5, final_beta, resolution)
        f_error = []
        f_error2 = []
        free_energy_V = []
        free_energy_T = []
        V_rate = []
        T_rate = []
        partition_T = []
        partition_V = []
        print('Operator ', self.__operator)
        for beta in beta_range:
            f_error.append( 1/np.sqrt(self.partition(beta, self.__T_evalues))* \
                      np.exp(0.5*beta*self.expectation(beta, self.__operator, self.__V_evalues, self.__U_V).real))
            f_error2.append(np.sqrt(self.partition(beta, self.__V_evalues)/self.partition(beta, self.__T_evalues)))
            partition_V.append(self.partition(beta, self.__V_evalues))
            partition_T.append(self.partition(beta, self.__T_evalues))
            free_energy_V.append(self.free_energy(beta, self.__V_evalues))
            free_energy_T.append(self.free_energy(beta, self.__T_evalues))
            V_rate.append(-beta*self.free_energy(beta, self.__V_evalues))
            T_rate.append(-beta*self.free_energy(beta, self.__T_evalues))
            print('Partition_T ', self.partition(beta, self.__T_evalues))
            print('Partition_V ', self.partition(beta, self.__V_evalues))
            print('Expectation in V ', self.expectation(beta, self.__operator, self.__V_evalues, self.__U_V))
        f_error = np.array(f_error)
        free_energy_V = np.array(free_energy_V)
        free_energy_T = np.array(free_energy_T)
        partition_T = np.array(partition_T)
        partition_V = np.array(partition_V)
        
        
        plt.figure(5)
        #plt.plot(beta_range, f_error, label='Fractional error')
        #plt.plot(beta_range, f_error2, label='Fractional error2')
        #plt.plot(beta_range, (T_rate), label='Growth rate T')
        #plt.plot(beta_range, (V_rate), label='Growth rate V')
        plt.plot(beta_range, (free_energy_V), label='Free energy V')
        plt.plot(beta_range, (free_energy_T), label='Free energy T')
        plt.plot(beta_range, 0.5*free_energy_V, label='0.5 Free energy V')
        #plt.plot(beta_range, np.sqrt(partition_V), label='Sqrt Partition V')
        plt.grid()
        plt.legend()
        
        plt.figure(6)
        plt.plot(beta_range, (np.sqrt(partition_V)/(partition_T)), label='Frac error')
        plt.grid()
        plt.legend()
        
        #plt.plot(beta_range, 0.5*free_energy_V-free_energy_T, label='Severity')

        #plt.plot(beta_range, n, label='n')
        #plt.plot(beta_range, f_error, label='Fractional error')

        
       
            
            
    def plot_severity_anal(self):
        
        beta_range = sp.linspace(0, 10, 50)
        s = []
        p = []
        n = []
        p_trial = 0
        n_trial = 0
        
        
        
        plt.figure(4)
        plt.plot(beta_range, s, label='Severity')
        #plt.plot(beta_range, p, label='p')
        #plt.plot(beta_range, n, label='n')
        plt.grid()
        plt.legend()
        plt.xlabel('Inverse Temperature [J]')
        plt.ylabel('Growth rate parameters')
        plt.savefig(str(self.__N)+'Frustrated_Severity_Analysis.jpeg')
        plt.show()
            
                
    
    def plot_evalues(self):
        plt.figure(1)
        plt.plot(self.__T_evalues, label='T eigenvalues')
        plt.plot(self.__V_evalues, label='V eigenvalues')
        plt.legend()
        plt.ylabel('Energy [-J]')
        plt.xlabel('Eigenstate index')
        plt.grid()
        plt.savefig(str(self.__N)+'Frustrated_SP_Analysis.jpeg')
        plt.show()
        
    def find_signal_strength(self):
        
        

        bins = sp.linspace(self.__V_evalues[0], self.__V_evalues[-1], 10)        
        pdf_x= histogram(self.__T_evalues, normed= True, bins= bins)[0]
        pdf_y= histogram(self.__V_evalues, normed= True, bins= bins)[0]
        #Transform the individual lines then multiply them followed by inverse transform
        c = np.fft.ifft(np.fft.fft(pdf_x)*np.fft.fft(pdf_y))
        c= c/c.sum()
        
        midpoints = []
        for i in range(len(bins)-1):
            midpoints.append(0.5*(bins[i]+bins[i+1]))
        
        plt.figure(3)
       # plt.plot(midpoints, c, label='T*V')
        plt.plot(midpoints, pdf_x, label='T')
        plt.plot(midpoints, pdf_y, label='V')
        plt.plot(midpoints, pdf_y-pdf_x, label='V-T')
        plt.xlabel('Energy [-J]')
        plt.ylabel('Normalised density of states')
        plt.grid()
        plt.legend()
        plt.savefig(str(self.__N)+'Frustrated_Signal_Analysis.jpeg')
        plt.show()
        
    def plot_DOS(self):   
        
        plt.figure(2)
        nT, binsT, patchesT = plt.hist(x=self.__T_evalues, bins=10, color='red',
                            alpha=0.7, rwidth=0.85, label='T')
                
        nV, binsV, patchesV = plt.hist(x=self.__V_evalues, bins=10, color='blue',
                            alpha=0.7, rwidth=0.85, label='V')
        plt.xlabel('Energy [-J]')
        plt.ylabel('Density of states')
        plt.legend()
        plt.savefig(str(self.__N)+'Frustrated_DOS_Analysis.jpeg')
        plt.show()

'''
Analyser = SP_Analysis(6, 1, 1)
Analyser.diagonalise()
Analyser.plot_evalues() 
#Analyser.plot_DOS()
#Analyser.find_signal_strength()
#Analyser.plot_severity_anal()
Analyser.bogoliubov_analysis(7, 700)
''' 
    
    
     

