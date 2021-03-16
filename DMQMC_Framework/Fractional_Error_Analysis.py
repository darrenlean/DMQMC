#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:39:06 2021

@author: darrenlean
"""

import numpy as np
import matplotlib.pyplot as plt

params = {
   'axes.labelsize': 10,
   'font.size': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [10, 5.25]
   } 
plt.rcParams.update(params)

class Fractional_Error:
    
    def __init__(self, N, H, D_rho):
        #Number of spins
        self.__N = N
        #The Hamiltonian object
        self.__H = H
        self.__H_matrix = H.Hamiltonian_matrix()
        
        self.__unique_entries, self.__counts = np.unique(self.__H_matrix, \
                                                         return_counts=True)
        self.__zero_entries_count = self.__counts[np.where(np.array(self.__unique_entries)==0)[0][0]]
        print('Hamiltonian D:', len(self.__H_matrix)-self.__zero_entries_count/len(self.__H_matrix))
        self.__D = min(D_rho, len(self.__H_matrix)-self.__zero_entries_count/len(self.__H_matrix))
        
        #Getting the smallest non-zero hamiltonian entry
        self.__H_min = max(abs(self.__unique_entries))
        for i in range(len(self.__unique_entries)):
            if abs(self.__unique_entries[i]) < self.__H_min and self.__unique_entries[i] != 0:
                self.__H_min = abs(self.__unique_entries[i])

    def get_plot_title(self, dbeta):
        '''
        This helper function make the title for plotting
        Naming convention follows the structure of the database:
        Type of Hamiltonian; Number of spins; dbeta; Number of initial psips
        '''
        
        title = ""
        #Removing the possible underscore in the name of the Hamiltonian
        for i in str(self.__H.__class__.__name__):
            if i == '_':
                title += " "
            else:
                title += i
        title += " "
        # Adding the number of spins
        title += "$N="+str(self.__N)+"$ "
        # Adding the finite difference step
        title += "$Δβ ="+str(dbeta)+"$ "
        
        self.__plot_title = title
        

    def f1max(self, R, dbeta):
        return 1/np.sqrt(R*self.__H_min*dbeta)

    def ftmax(self, R, dbeta):
        a = 2 + self.__H_min * dbeta
        b = -self.__D/R
        c = b
        discriminant = np.sqrt(b**2 - 4*a*c)
        return (-b + discriminant)/(2*a)
    
    def f1maxGraph(self, Rmax, dbeta):
        
        #Making the plot title
        self.get_plot_title(dbeta)
        
        all_R = []
        all_f1max = []
        
        for i in range(1, Rmax+1):
            all_R.append(i)
            all_f1max.append(self.f1max(i, dbeta))
        
        plt.xlabel('Initial population on diagonal entry')
        plt.ylabel(r'$f$')
        plt.plot(all_R, all_f1max, label = 'Theoretical $f^{(1)}_{max}$')
        plt.plot([all_R[0], all_R[-1]], [1, 1], label = '1')
        plt.legend()
        plt.grid()
        plt.title(self.__plot_title)
        
    def ftmaxGraph(self, Rmax, dbeta):
        
        #Making the plot title
        self.get_plot_title(dbeta)
        
        all_R = []
        all_ftmax = []
        
        for i in range(1, Rmax+1):
            all_R.append(i)
            all_ftmax.append(self.ftmax(i, dbeta))
        
        plt.xlabel('Initial population on diagonal entry')
        plt.ylabel(r'$f$')
        plt.plot(all_R, all_ftmax, label = r'Theoretical $f^{(t)}_{max}$')
        plt.plot([all_R[0], all_R[-1]], [1, 1], label = '1')
        plt.legend()
        plt.grid()
        plt.title(self.__plot_title)
        
    def fGraph(self, Rmax, dbeta):
        
        #Making the plot title
        self.get_plot_title(dbeta)
        
        all_R = []
        all_f1max = []
        all_ftmax = []
        
        for i in range(1, Rmax+1):
            all_R.append(i)
            all_f1max.append(self.f1max(i, dbeta))
            all_ftmax.append(self.ftmax(i, dbeta))
        
        #plt.xlabel('Initial population on diagonal entry')
        plt.xlabel(r'$R$')
        plt.ylabel(r'$f$')
        plt.plot(all_R, all_f1max, label = 'Theoretical $f^{(1)}_{max}$')
        plt.plot(all_R, all_ftmax, label = r'Theoretical $f^{(t)}_{max}$')
        #plt.plot([all_R[0], all_R[-1]], [1, 1], label = '1')
        plt.legend()
        plt.grid()
        #plt.title(self.__plot_title)