#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:00:19 2021

@author: darrenlean
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import ExactDiagonalisation as ED
import Sign_Problem_Analysis as sign_solver
from scipy.optimize import curve_fit

params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [10, 5.25]
   } 
plt.rcParams.update(params)

class Finite_Beta_Analyser:
    '''
    Diagnostic tool for a DMQMC simulation at finite temperature estimations
    '''
    
    def __init__(self, Hamiltonian, N, dbeta, init_pop):
        '''
        The required inputs are necessary for navigating the database, which 
        has a structure of Database/Type_of_Hamiltonian/Number_of_spins/dbeta/
        Number_of_initial_psips/Loop_index/(all the raw data text files)
        '''
        #The Hamiltonian object
        self.__H = Hamiltonian
        #The number of spins
        self.__N = N
        #Finite difference step
        self.__dbeta = dbeta
        #Initial number of psips on each of the diagonal entries
        self.__init_pop = init_pop
        #Naming convention follows the structure of the database:
        #Type of Hamiltonian; Number of spins; dbeta; Number of initial psips
        self.__folder_name = str(self.__H.__class__.__name__)+'/'+\
                'N='+str(N)+'/dbeta='+str(dbeta)+'/initpop='+str(init_pop)+'/'
        #Directory to navigate through the database
        self.__data_directory = 'Database/' + self.__folder_name
        #Getting the total number of loops for this particular folder
        self.__total_no_loops = max(self.get_all_loop_indices())
        #Printing the status of the folder
        print(self.__data_directory +': '+str(self.__total_no_loops)+'loops')
        
        
        #All memory allocation
        self.__beta = []
        self.__beta_error = self.__dbeta/2
        
        self.__no_psips = []
        self.__no_psips_estimators = []
        self.__no_psips_errors = []
        self.__psips_growth = []
        self.__psips_growth_errors = []
        
        self.__rho_trace = []
        self.__H_trace = []
        self.__energy_estimators = []
        self.__energy_errors = []
        self.__shifts = []
        
        # only need to load target beta once
        self.load_target_betas()
    
    def directory_check(self, directory):
        '''
        This helper function creates the folder of the given directory if it
        is not found
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def get_all_loop_indices(self):
        '''
        This helper function retrieve all the loop indices
        '''
        folder_names = []
        for entry_name in os.listdir(self.__data_directory):
            entry_path = os.path.join(self.__data_directory, entry_name)
            if os.path.isdir(entry_path):
                folder_names.append(int(entry_name))
        
        return folder_names
    
    def textfile_finder(self, directory):
        '''
        This helper function that returns all the files ending with '.txt'
        in the given directory
        '''
        fileNames = []
        for file in os.listdir(directory+'/'):
            if file.endswith(".txt"):
                fileNames.append(file)
        return fileNames    
    
    def load_target_betas(self):
        '''
        This helper function loads the target betas from the first loop found
        '''
        
        current_directory = self.__data_directory+str(self.get_all_loop_indices()[0])+'/'
        #Getting the raw data text file names
        all_textfile_names = self.textfile_finder(current_directory)
        
        if 'target_betas.txt' in all_textfile_names:
            with open(current_directory + 'target_betas.txt', 'r') as file_object:
                for line in file_object:
                    parts = line.split()
                    self.__beta.append(float(parts[0]))
        else:
            #Raising this because it is an essential data
            raise ("Target betas do not exist!")
    
    def load_loop(self, loop_index):
        '''
        This helper function loads data for a selected loop
        '''
        
        #Taking the directory into the selected loop folder
        current_directory = self.__data_directory+str(loop_index)+'/'
        #Getting the raw data text file names
        all_textfile_names = self.textfile_finder(current_directory)
            
        if 'rho_traces.txt' in all_textfile_names:
            with open(current_directory + 'rho_traces.txt', 'r') as file_object:
                temp = []
                for line in file_object:
                    parts = line.split()
                    temp.append(float(parts[0]))
                self.__rho_trace.append(temp)
        else:
            #Raising this because it is an essential data
            raise ("Rho traces do not exist!")
            
        if 'H_rho_traces.txt' in all_textfile_names:
            with open(current_directory + 'H_rho_traces.txt', 'r') as file_object:
                temp = []
                for line in file_object:
                    parts = line.split()
                    temp.append(float(parts[0]))
                self.__H_trace.append(temp)
        else:
            #Raising this because it is an essential data
            raise ("H rho traces do not exist!")
            
        if 'total_psips.txt' in all_textfile_names:
            with open(current_directory + 'total_psips.txt', 'r') as file_object:
                temp = []
                for line in file_object:
                    parts = line.split()
                    temp.append(float(parts[0]))
                self.__no_psips.append(temp)
        else:
            #Raising this because it is an essential data
            raise ("Total populations do not exist!")
            
        if 'shifts.txt' in all_textfile_names:
            with open(current_directory + 'shifts.txt', 'r') as file_object:
                temp = []
                for line in file_object:
                    parts = line.split()
                    temp.append(float(parts[0]))
                self.__shifts.append(temp)
        else:
            #Raising this because it is an essential data
            raise ("shifts do not exist!")
            
    def load_all_data(self):
        '''
        This helper function uses load_data to load data from all loops
        '''
        for loop_index in self.get_all_loop_indices():
            self.load_loop(loop_index)
            
    def population_analysis(self):
        '''
        Performing averaging of population, reconstructing the growth of 
        population and the associated error analysis
        '''

        #Loop through the various target betas
        for i in range(len(self.__beta)):
            p_array = []
            shift_array = []
            #Loop through the various loops to get data from all loops at fixed beta
            for loop_index in range(self.__total_no_loops):
                p_array.append(self.__no_psips[loop_index][i])
                shift_array.append(self.__shifts[loop_index][i])
            
            #Population averaging and error
            p = np.mean(p_array)
            p_err = np.std(p_array, ddof=1)/np.sqrt(len(p_array))
            self.__no_psips_estimators.append(p)
            self.__no_psips_errors.append(p_err)
            
            #Population growth reconstruction and error
            exp = np.exp(-np.mean(shift_array)*self.__beta[i])
            y = p*exp
            shift_err = np.std(shift_array, ddof=1)/np.sqrt(len(shift_array))
            frac1 = exp*p_err/p
            frac2 = p*self.__beta[i]*exp*shift_err/np.mean(shift_array)
            self.__psips_growth.append(y)
            self.__psips_growth_errors.append(y*np.sqrt(frac1**2 + frac2**2))
            
         
    def growth_fitting(self):
        '''
        This helper function fits the reconstructed population in log space and
        returns the fit parameters in log space 
        (log population = fit[0]*log beta + fit[1])
        '''
        log_psips_growth = np.log(self.__psips_growth)
        log_psips_growth_err = []
        for i  in range(len(self.__no_psips_errors)):
            log_psips_growth_err.append(self.__psips_growth_errors[i]*np.exp(1)/\
                        (self.__psips_growth[i]*np.log(self.__psips_growth[i])))
        #Guessing the gradient and intercept
        guess = [(log_psips_growth[-1]-log_psips_growth[0])/self.__beta[-1],\
                 log_psips_growth[0]]
        #Curve fitting for the log of the reconstructed population
        #curve_fit(self.linear, self.__beta, log_psips_growth, p0=guess, \
        #          sigma = log_psips_growth_err)
        fit_parameters, fit_error_cov = \
        curve_fit(self.linear, self.__beta, log_psips_growth, p0=guess)
        print('Fitted growth rate: ', fit_parameters[0], ' +/- ', \
              np.sqrt(np.diag(np.real(fit_error_cov)))[0])
        
        self.__growth_rate = fit_parameters[0]
        self.__growth_rate_error = np.sqrt(np.diag(np.real(fit_error_cov)))[0]
        
        self.__growth_init = fit_parameters[1]
        self.__growth_init_error = np.sqrt(np.diag(np.real(fit_error_cov)))[1]
        
        return fit_parameters, np.sqrt(np.diag(np.real(fit_error_cov)))
        
            
    def linear(self, x, m, c):
        '''
        Linear function for fitting
        '''
        return m*x+c
    
    def energy_analysis(self):
        '''
        Performing local energy estimations
        '''
        #Loop through the various target betas
        for i in range(len(self.__beta)):
            rho_trace_array = []
            H_trace_array = []
            #Loop through the various loops to get data from all loops at fixed beta
            for loop_index in range(self.__total_no_loops):
                rho_trace_array.append(self.__rho_trace[loop_index][i])
                H_trace_array.append(self.__H_trace[loop_index][i])
            
            measures = np.array([rho_trace_array, H_trace_array])
                
            mean_proj = np.mean(measures[1])
            mean_trace = np.mean(measures[0])
            
            for i in range(len(measures[0])):
                measures[1][i] /= measures[0][i]
        
            std_error_proj = np.std(measures[1], ddof=1)#\
                           # /np.sqrt(len(measures[1]))
            std_error_trace = np.std(measures[0], ddof=1)#\
                            #/np.sqrt(len(measures[0]))
        
            mean_energy = mean_proj/mean_trace
            std_error_energy = abs(mean_energy)\
                                *(np.sqrt((std_error_proj/mean_proj)**2+\
                                  (std_error_trace/mean_trace)**2-\
                                  (2*np.cov(measures)[0][1])/\
                                  abs((mean_proj*mean_trace))))

            self.__energy_estimators.append(mean_energy)
            self.__energy_errors.append(std_error_proj/np.sqrt(len(measures[0])))
            
    def get_plot_title(self):
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
        title += "$Δβ ="+str(self.__dbeta)+"$ "
        # Adding the number of initial psips on each of the diagonal entry
        title += "$n_0="+str(self.__init_pop)+"$ "
        
        self.__plot_title = title
            
    def run_diagnostics(self):
        '''
        This helper function plots the graphs
        '''
        
        #Making the plot title
        self.get_plot_title()
        #Directory to navigate through the result folder
        self.__result_directory = 'Results/' + self.__folder_name
        #Check if the result directory exists. If not, create it
        self.directory_check(self.__result_directory)
        
        self.growth_fitting()
        
        ED_energy = \
        ED.ED(self.__N, self.__beta[-1], 2*len(self.__beta), self.__H.Hamiltonian_matrix())
        
        for i  in range(len(self.__no_psips_errors)):
            self.__no_psips_errors[i] *= np.exp(1)/(self.__no_psips_estimators[i]\
                                  *np.log(self.__no_psips_estimators[i]))
        
        plt.figure()
        plt.title(self.__plot_title)
        plt.errorbar(self.__beta, np.log(self.__no_psips_estimators), \
                     xerr = (self.__dbeta/2)*np.ones(len(self.__beta)), \
                     yerr = np.array(self.__no_psips_errors), fmt = 'x', label='DMQMC')
        plt.xlabel(r'$\beta$ [J]')
        plt.ylabel('Log of total psip population')
        plt.grid()
        plt.savefig(fname = self.__result_directory + "Original_psip" + ".jpeg", format = 'jpeg')
        plt.show()
        
        for i  in range(len(self.__no_psips_errors)):
            self.__psips_growth_errors[i] *= np.exp(1)/(self.__psips_growth[i]\
                                      *np.log(self.__psips_growth[i]))
            
        plt.figure()
        plt.title(self.__plot_title)
        plt.plot(self.__beta, np.log(self.__psips_growth),'x', label='DMQMC') \
                     #xerr = (self.__dbeta/2)*np.ones(len(self.__beta)), \
                     #yerr = np.array(self.__psips_growth_errors), fmt = 'x', label='DMQMC')
        
        
        #plt.errorbar(self.__beta, np.log(self.__psips_growth), \
        #             xerr = (self.__dbeta/2)*np.ones(len(self.__beta)), \
        #             yerr = np.array(self.__psips_growth_errors), fmt = 'x', label='DMQMC')
        plt.plot(self.__beta, self.linear(np.array(self.__beta), self.__growth_rate, \
                                          self.__growth_init), label='Fit')
        plt.xlabel(r'$\beta$ [J]')
        plt.ylabel('Log of total recon. psip population growth')
        plt.legend()
        plt.grid()
        plt.savefig(fname = self.__result_directory + "Reconstructed_psip" + ".jpeg", format = 'jpeg')
        plt.show()
        
        
        plt.figure()
        plt.errorbar(self.__beta, self.__energy_estimators, \
                     xerr = (self.__dbeta/2)*np.ones(len(self.__beta)), \
                     yerr = self.__energy_errors, fmt = 'x', label='DMQMC')
        
        plt.title(self.__plot_title)
        plt.xlabel(r'$\beta$ [J]')
        plt.ylabel('Energy estimator [J]')
        ED_energy.show_result()
        plt.grid()

        plt.savefig(fname = self.__result_directory + "Energy_profile" + ".jpeg", format = 'jpeg')
        plt.show()
        
        #plt.figure(4)
        #plt.plot(self.__beta, self.__energy_errors)
        
class Growth_Analyser:
    '''
    Diagnostic tool for a DMQMC growth rate across different initial population
    It requires Finite_Beta_Analyser
    '''
    def __init__(self, Hamiltonian, N, dbeta):
        #The Hamiltonian object
        self.__H = Hamiltonian
        #The number of spins
        self.__N = N
        #Finite difference step
        self.__dbeta = dbeta
        #Naming convention follows the structure of the database:
        #Type of Hamiltonian; Number of spins; dbeta; Number of initial psips
        self.__folder_name = str(self.__H.__class__.__name__)+'/'+\
                'N='+str(N)+'/dbeta='+str(dbeta)+'/'
        #Directory to navigate through the database
        self.__data_directory = 'Database/' + self.__folder_name
        self.__all_init_pop = self.get_init_pop()
        self.__all_init_pop.sort()
        
        #Obtaining the Tmax and Vmax of the given hamiltonian
        self.__Tmax, self.__Vmax = sign_solver.TmaxVmax(self.__H.Hamiltonian_matrix())
       
    def directory_check(self, directory):
        '''
        This helper function creates the folder of the given directory if it
        is not found
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    def get_init_pop(self):
        '''
        This helper function retrieve all the initial population folder names
        '''
        folder_names = []
        for entry_name in os.listdir(self.__data_directory):
            entry_path = os.path.join(self.__data_directory, entry_name)
            if os.path.isdir(entry_path):
                folder_names.append(int(entry_name[8:]))
        return folder_names
    
    def get_all_fittings(self):
        '''
        This function uses Finite_Beta_Analyser to get all the fittings for 
        different initial population
        '''
        self.__rates = []
        self.__rates_error = []
        for init_pop in self.__all_init_pop:
            FBA = Finite_Beta_Analyser(self.__H, self.__N, self.__dbeta, init_pop)
            FBA.load_all_data()
            FBA.population_analysis()
            fit_parameters, fit_errors = FBA.growth_fitting()
            #FBA.run_diagnostics()
            self.__rates.append(fit_parameters[0])
            self.__rates_error.append(fit_errors[0])
        print(self.__rates)
            
    def rate_fitting(self):
        '''
        This helper function fits the growth rate and returns the fit parameters
        '''
        self.__crit_index = np.where(np.array(self.__rates)==min(self.__rates))[0][0]
        #Guessing the gradient and intercept
        guess = [(self.__rates[self.__crit_index]-self.__rates[0])/\
                 (self.__all_init_pop[self.__crit_index]-self.__all_init_pop[0]), \
                 self.__rates[0]]
        #Curve fitting for the log of the reconstructed population
        fit_parameters, fit_error_cov = \
        curve_fit(self.linear, self.__all_init_pop[:self.__crit_index], \
                  self.__rates[:self.__crit_index], p0=guess, \
                  sigma = self.__rates_error[:self.__crit_index])
        
        print('Fitted rate gradient: ', fit_parameters[0], ' +/- ', \
              np.sqrt(np.diag(np.real(fit_error_cov)))[0])
        
        self.__rate_gradient = fit_parameters[0]
        self.__rate__gradient_error = np.sqrt(np.diag(np.real(fit_error_cov)))[0]
        
        self.__rate_init = fit_parameters[1]
        self.__rate_init_error = np.sqrt(np.diag(np.real(fit_error_cov)))[1]
        
        return fit_parameters, np.sqrt(np.diag(np.real(fit_error_cov)))
        
            
    def get_plot_title(self):
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
        title += "$Δβ ="+str(self.__dbeta)+"$ "
        
        self.__plot_title = title
        
    def linear(self, x, m, c):
        '''
        Linear function for fitting
        '''
        return m*x+c
    
    def run_diagnostics(self):
        '''
        This helper function plots the graphs
        '''
        #Making the plot title
        self.get_plot_title()
        #Directory to navigate through the result folder
        self.__result_directory = 'Results/' + self.__folder_name
        #Check if the result directory exists. If not, create it
        self.directory_check(self.__result_directory)
        
        plt.figure()
        plt.errorbar(self.__all_init_pop, self.__rates, \
                     yerr = self.__rates_error, fmt = 'x', label='DMQMC')
        plt.plot(self.__all_init_pop[:self.__crit_index], \
                 self.linear(np.array(self.__all_init_pop[:self.__crit_index]),\
                  self.__rate_gradient, self.__rate_init), label='Fit')
        plt.plot([self.__all_init_pop[0], self.__all_init_pop[-1]], \
                 [self.__Tmax, self.__Tmax], 'g--', label=r'$T_{max}$')
        plt.plot([self.__all_init_pop[0], self.__all_init_pop[-1]], \
                 [self.__Vmax, self.__Vmax], 'r--', label=r'$V_{max}$')
        plt.xlabel("Initial population on each of diagonal entry")
        plt.ylabel("Growth rate (fitted exponentials)")
        plt.title(self.__plot_title)
        plt.legend()
        plt.grid()
        plt.savefig(fname = self.__result_directory + "Growth_rate" + ".jpeg", format = 'jpeg')