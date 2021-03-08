#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMQMC Core Algorithm Implementation 
"""
import os
import numpy as np
import matplotlib.pyplot as plt

class DMQMC_Core:
    def __init__(self, N, Hamiltonian, dbeta, init_pop=10):
        
        #Number of spins
        self.__N = N
        # Dimension of Hilbert space
        self.__dimension = int(2**N)
        # Number of psips in the simulation, updated on the fly
        self.__total_no_psips = 0
        
        # Storage unit of occupied density matrix element
        self.__rho = {}
        # Storage dictionary for newly spawned psip populations
        self.__rho_temp = {}
        #Initial population on each on the diagonal entry
        self.__init_pop = init_pop
        
        # Inverse temperature unit to convert matrix entries to probabilities
        self.__dbeta = dbeta
        self.__betasteps = 0
        # Shift estimator for energy eigenvalue during simulation
        self.__shift = 0
        
        # Local energy estimator calculated from density matrix projection
        self.__local_energy = 0
        # Store the current trace of density matrix
        self.__trace = 0
        self.__H_trace = 0
        
        # Index for output file names for different estimators and data
        self.__output_files = []
        
        # Hamiltonian imported from Hamiltonians generator class        
        self.__Hamiltonian_handle = Hamiltonian
        
        #Starting the directory by looking into the database
        self.__directory = 'Database'
        #The first level is the name of the Hamiltonian
        self.__directory += '/'+str(self.__Hamiltonian_handle.__class__.__name__)
        #***Still haven't found a good way to store the coupling strengths***
        #The second level is the number of spins
        self.__directory += '/N='+str(self.__N)
        #The third level is the finite difference beta step
        self.__directory += '/dbeta='+str(self.__dbeta)
        #The four level is the initial population
        self.__directory += '/initpop='+str(self.__init_pop)+'/'
        #Checking if the directory exists. If not, create one
        self.directory_check(self.__directory)
        
        
    def directory_check(self, directory):
        '''
        This helper function creates the folder of the given directory if it
        is not found
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def open_output(self, quantity):
        '''
        Create an output stream for some quantity and append it with the number 
        of particles and the applied shift
        '''
        
        self.__output_files.append(open(self.__directory+quantity+'.txt', 'a+'))


        
    def close_outputs(self):
        '''
        Close all output streams
        '''
        for output in self.__output_files:
            output.close()
        self.__output_files = []
       
    def no_psips(self):
        '''
        Count total number of psips in simulations
        '''
        summa = 0
        for site in self.__rho:
            # Psip populations are signed on each entry
            summa += abs(self.__rho[site])
        return summa
    
    def connected_sites(self, site):
        '''
        Return connected indices for some many-body basis state (site)
        '''
        return self.__Hamiltonian_handle.connected_sites(site)
        
    def hamiltonian(self, row_index, col_index):
        '''
        Generate entry of the Hamiltonian matrix in the many-body basis
        '''
        
        return self.__Hamiltonian_handle.Hamiltonian(row_index, col_index)
    
    
    def set_shift(self, init_shift):
        '''
        Set simulation shift to some value
        '''
        self.__shift = init_shift
    
    def trans_matrix_elem(self, row_index, col_index):
        '''
        Generate transition matrix at some entry of the Hamiltonian matrix
        '''
        if (row_index == col_index):
            return -(self.hamiltonian(row_index, col_index) - self.__shift)
        else:
            return -self.hamiltonian(row_index, col_index)
    
    def sign(self, i):
        '''
        Sign of a psip population
        '''
        if (i!=0):
            return int(i/abs(i))
        else:
            return 0
     

    def iterate(self):
        
        def attempt_spawn_col(row_index, col_index, target_index, pop_sign, T_elem):
            '''
            Attempt spawning at fixed row, along the coloumns of the density 
            matrix for some target coloumn index
            For convenience, pass on the sign of the parent psip population
            '''
            
            # Avoid self-spawning onto the parent population
            if (target_index == row_index):
                return
            
            else:
                
                # Spawning probability is detemined from transition matrix entry
                #trans_matrix_elem = self.trans_matrix_elem(row_index, target_index)          
                prob = 0.5*abs(T_elem)*self.__dbeta
                
                if (np.random.uniform() < prob):
                    new_psip_sign = self.sign(T_elem)*pop_sign
                    
                    # Find the storage index of the target entry
                    #result_index = self.find_entry(target_index, col_index)
                    # If entry on which to spawn is occupied, store new psip
                    # in a temporary population
                    if ((target_index, col_index) in self.__rho_temp):
                        self.__rho_temp[target_index, col_index] += new_psip_sign
                    # If density matrix entry is unoccupied, create a new occupied
                    # storage unit
                    else:
                        # Update temprary population of the newly appended unit
                        self.__rho_temp[target_index, col_index] = new_psip_sign
   
                
        def attempt_spawn_row(row_index, col_index, target_index, pop_sign, T_elem):
            '''
            Attempt spawning at fixed coloumn, along the rows of the density 
            matrix for some target row index
            For convenience, pass on the sign of the parent psip population
            Process is exactly analogous to the previous one
            '''
            
            if (target_index == col_index):
                return
            
            else:
                
                #trans_matrix_elem = self.trans_matrix_elem(target_index, col_index)               
                prob = 0.5*abs(T_elem)*self.__dbeta
                
                if (np.random.uniform() < prob):
                    new_psip_sign = self.sign(T_elem)*pop_sign
                    
                    if ((row_index, target_index) in self.__rho_temp):
                        self.__rho_temp[row_index, target_index] += new_psip_sign
                    # If density matrix entry is unoccupied, create a new occupied
                    # storage unit
                    else:
                        # Update temprary population of the newly appended unit
                        self.__rho_temp[row_index, target_index] = new_psip_sign       

            
        def attempt_diag(site_index, pop_sign, T_elem):
            '''
            Attempt DMQMC diagonal step for some density matrix entry
            site_index is the storage index of the population attempting the 
            step
            '''

            # Note probability is the absolute value of the sum of transition
            # matrix entries!
            prob = 0.5*abs(T_elem)*self.__dbeta
            # If trans_matrix_elems > 0 -> cloning 
            # If trans_matrix_elems < 0 -> death
            # These steps are simultaneously implemented by allowing the 
            # new psip's sign to be either opposite or equal to the current 
            # population
            
            if (np.random.uniform() < prob):
                new_psip_sign = self.sign(T_elem)*pop_sign
            
                if (site_index in self.__rho_temp):
                    self.__rho_temp[site_index] += new_psip_sign
                    # If density matrix entry is unoccupied, create a new occupied
                    # storage unit
                else:
                    # Update temprary population of the newly appended unit
                    self.__rho_temp[site_index] = new_psip_sign   
                    


        # On and off-diagonal spawning dynamics
        # Loop over all parent sites
        
        for site_index in self.__rho:
            
            occup_site = site_index
            
            # Access indices and signed population of density matrix entry
            row_index = occup_site[0]
            col_index = occup_site[1]
            pop = self.__rho[occup_site]
            pop_sign = self.sign(pop)
            connected_rows = self.connected_sites(col_index)
            connected_cols = self.connected_sites(row_index)
            
            connected_rows_T = []
            for i in range(len(connected_rows)):
                connected_rows_T.append(self.trans_matrix_elem(connected_rows[i], col_index))
            
            connected_cols_T = []
            for i in range(len(connected_cols)):
                connected_cols_T.append(self.trans_matrix_elem(row_index, connected_cols[i]))
                
            diag_T = self.trans_matrix_elem(site_index[0], site_index[0])+\
            self.trans_matrix_elem(site_index[1], site_index[1])
            
                
            # Loop over every psip in population
            for psip in range(abs(self.__rho[occup_site])):
                
                # Attempt spawning along rows
                # NEEDS OPTIMISATION
                for i in range(len(connected_rows)):
                    attempt_spawn_row(row_index, col_index, connected_rows[i],\
                                      pop_sign, connected_rows_T[i])
                    
                # Attempt spawning along coloumns
                # NEEDS OPTIMISATION
                for i in range(len(connected_cols)):
                    attempt_spawn_col(row_index, col_index, connected_cols[i],\
                                      pop_sign, connected_cols_T[i])
                    
                # Attempt diagonal for entry:
                attempt_diag(site_index, pop_sign, diag_T)
                
       
        
                                
        # Every loop, trace may change so set it to zero at start
        self.__trace = 0
        # Calculate the estimate for Tr[rho*H]
        self.__H_trace = 0
        
        
        # Annihilation step
        summa = 0       
 
        for site in self.__rho_temp:
            # Carry out annihilation on every non-empty entry
            # by merging temporary signed populations with the parent ones
            if site in self.__rho:
                self.__rho[site] += self.__rho_temp[site]
            else:
                self.__rho[site] = self.__rho_temp[site]
                
        # Empty temporary population
        self.__rho_temp = {}
    
    
        for site in self.__rho:
        
            # Keep updating the total number of psips
            summa += abs(self.__rho[site])
            
            # For local energy estimation, calculate Tr[rho*H] on the fly
            self.__H_trace += self.__rho[site]*self.hamiltonian(site[1], site[0])
                        
            # Keep updating the trace of density matrix
            if (site[0] == site[1]):
                self.__trace += self.__rho[site]
                            
        # Calculate local energy estimator in the loop
        self.__local_energy = self.__H_trace/self.__trace
            
        # Collect diagnostics from simulation
        self.__total_no_psips = summa

            
    def run_diagnostics(self):
        
        no_betasteps = len(self.__no_psips_evol)
        betasteps = np.linspace(0, no_betasteps, no_betasteps)
        '''
        print('---Run diagnostics---')
        print('----------------------')
        print('Simulation Parameters')
        print('Beta step: ')
        print(self.__dbeta)    
        print('No. betasteps')
        print(no_betasteps)
        print('Final shift')
        print(self.__shift)
        print('No. psips')
        print(self.__total_no_psips)
        print('Final, norm. density matrix')
        print(self.normalise_density_matrix())
        print('Stable index')
        print(self.__stable_index)
        '''
        plt.figure(1)
        #plt.title('Evolution of total psip population')
        plt.plot(betasteps, np.log10(self.__no_psips_evol), 'x')
        plt.xlabel(r'$\beta$')
        plt.ylabel('Log of Total number of psips')
        plt.grid()
        
        plt.figure(2)
        plt.plot(betasteps, self.__local_energy_evol, 'x')
        plt.xlabel(r'$\beta$')
        plt.ylabel('Local energy estimator')
        plt.grid()
        
        
    def construct_density_matrix(self):
        '''
        For small dimensions, one can explicitly construct the density matrix
        for benchmarking
        DON'T USE FOR LARGE SYSTEMS
        '''
        
        if (self.__N > 4):
            print('WARNING: memory overload of storing density matrix')
        
        density_matrix = np.zeros([self.__dimension, self.__dimension])
        
        for occup_site in self.__rho:
            
            density_matrix[occup_site] = self.__rho[occup_site]
        
        return density_matrix
    
    def normalise_density_matrix(self):
        '''
        Normalise density matrix st Tr[rho] = 1
        '''
        
        return self.construct_density_matrix()/self.__trace
    
    def update_shift(self, m, earlier_pop):
        '''
        Adjusting shift factor based on sign change and ratio of total 
        population
        '''
        alpha = 0.1
        #population ratio = 1 - shift*dtau + eigenvalue*dtau
        #we want shift = eigenvalue
        #so shift = (ratio - 1)/dtau + shift

        self.__shift -= (alpha/(m*self.__dbeta))*np.log(self.__total_no_psips\
                         /earlier_pop)

    

    def run_beta_loop(self, target_beta, dbeta, no_loops):
        
        max_no_psips = 1000000

        
        self.__dbeta = dbeta
        betasteps = int(target_beta/dbeta)
        
        # Reassign psip populations to beta = 0 (identity)
        
        m = 1
        earlier_pop = 0
        
        '''
        if no_loops%4 == 0:
            init_no_occup_sites = int(self.__dimension/4)
            chosen_sites = np.random.randint(0, self.__dimension, init_no_occup_sites)
        if no_loops == 1:
            init_no_occup_sites = self.__dimension
            chosen_sites = np.linspace(0, self.__dimension-1, self.__dimension)
        '''
        init_no_occup_sites = self.__dimension
        chosen_sites = np.linspace(0, self.__dimension-1, self.__dimension)
        
        self.__rho = {}
        self.__rho_temp = {}
        self.__total_no_psips = 0
        
        for i in range(init_no_occup_sites):
            self.__rho[(int(chosen_sites[i]), int(chosen_sites[i]))] = self.__init_pop
            self.__total_no_psips += self.__init_pop        
            
        earlier_pop = self.__total_no_psips
        
        step_index = 1
        while(step_index < betasteps+1 and self.__total_no_psips <  max_no_psips):
                
            self.iterate()
            if (step_index%m==0):
                self.update_shift(m, earlier_pop)
                earlier_pop = self.__total_no_psips
            
                
           # if (step_index%11 == 0):
            '''
            print('Target beta ', target_beta)
            print('Beta: ', step_index*self.__dbeta)
            print('Loop completion (%): ', 100*(step_index-1)/betasteps)
            print('Total psip population ', self.__total_no_psips)
            print('Current shift: ', self.__shift)
            print('Local energy estimator', self.__local_energy)
            '''
            # Calculate local energy estimator in the loop
            self.__local_energy = self.__H_trace/self.__trace
            # Collect diagnostics from beta loop

            step_index += 1
            


        
    
    def run_finite_beta(self, no_loops, final_beta, resolution):
        '''
        Run 'no_loops' finite beta loops to estimate finite temperature 
        properties in the range of betas [0, final_beta] at some resolution 
        (ie. number of beta-s for collecting diagnostics). 
        Betasteps is 1/dbeta, i.e. the number of betasteps in the unit
        '''
        
        if no_loops != 1 and no_loops%4 != 0:
            raise ValueError("Number of loops must be either 1 or a multiple of 4!")
        
        
        target_beta_range = np.linspace(0, final_beta, resolution+1)
        
           
        for target_beta in target_beta_range:
            # Skip infinite temperature (all properties are known)
            if (target_beta==0):
                continue
            for loop_index in range(1, no_loops+1):
                self.directory_check(self.__directory+str(loop_index)+'/')
                self.open_output(str(loop_index)+'/target_betas') # index 0
                self.open_output(str(loop_index)+'/rho_traces') # index 1
                self.open_output(str(loop_index)+'/H_rho_traces')  # index 2
                self.open_output(str(loop_index)+'/total_psips') #index 3
                self.open_output(str(loop_index)+'/shifts') # index 4
                self.open_output(str(loop_index)+'/no_occup_sites') # index 5
                
                self.run_beta_loop(target_beta, self.__dbeta, no_loops)
                print()
                print('Loop number ', loop_index, ' out of ', no_loops)
                print('Current target beta: ', target_beta, 'Final target beta:', final_beta)
                print('Total psip pop. ', self.__total_no_psips)
                print('Shift: ', self.__shift)
                print('Local energy: ', self.__local_energy)
                self.__output_files[0].write(str(target_beta)+'\n')
                self.__output_files[1].write(str(self.__trace)+'\n')
                self.__output_files[2].write(str(self.__H_trace)+'\n')
                self.__output_files[3].write(str(self.__total_no_psips)+'\n')
                self.__output_files[4].write(str(self.__shift)+'\n')
                self.__output_files[5].write(str(len(self.__rho))+'\n')
            
                self.close_outputs()
