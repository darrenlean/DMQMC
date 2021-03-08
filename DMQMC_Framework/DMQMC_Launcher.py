#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for launching DMQMC algorithms
"""
import time
import ExactDiagonalisation as ED
import Hamiltonians as H_generator
import DMQMC_Analysis as Analysis
import Sign_Problem_Analysis as sign_solver
import Fractional_Error_Analysis as FEA
import DMQMC_Core as Simulator

#%%

# Running an exact diagonalisation for finding ground state

N = 6
J2 = 1
J1 = 1
final_beta = 12
resolution = 40
betasteps=20

H = H_generator.Frustrated_Ladder(N,J1,J2)

H_matrix = H.Hamiltonian_matrix()

startTime = time.time() 

test = ED.ED(N, final_beta, resolution, H_matrix)   
test.show_result()

#test.show_hamiltonian()

endTime = time.time()
print('Total time taken', (endTime-startTime)/60, 'minutes')

#%%
# Running sign problem analysis

N = 8
J2 = 1
J1 = 1
final_beta = 7
resolution = 20
betasteps=8

H = H_generator.Frustrated_Ladder(N, J1, J2)
H_matrix = H.Hamiltonian_matrix()
sign_solver.TmaxVmax(H_matrix)
print(sign_solver.TmaxVmax(H_matrix))

#%%

# One type initial population 
# Running a diagnostic tools
# ***Input parameters here***
# Number of spins
N = 8
# Coupling strength
J1 = 1
J2 = 1
# Finite difference beta step
dbeta = 1/8
# Initial population on each of the diagonal entry
init_pop = 30


# ***Select the Hamiltonian here***
H = H_generator.Frustrated_Ladder(N, J1, J2)
 
startTime = time.time() 

analyser = Analysis.Finite_Beta_Analyser(H, N, dbeta, init_pop)

analyser.load_all_data()
analyser.population_analysis()
analyser.energy_analysis()
analyser.run_diagnostics()


endTime = time.time()
print('Total time taken', (endTime-startTime)/60, 'minutes')
        
#%%

# Running DMQMC

# ***Input parameters here***
# Number of spins
N = 10
# Coupling strength
J1 = 1
J2 = 1
# Initial population on each of the diagonal entry
init_pop = 10
# Number of loops
no_loops = 1
# Final target beta
final_beta = 6
# final_beta/resolution = integer * dbeta
# resolution = final_beta/(integer*dbeta)
# Finite difference beta step
dbeta = 1/8
# Number of measurements to be made between beta = 0 and beta = final_beta
resolution = final_beta/(2*dbeta)

# ***Select the Hamiltonian here***
H = H_generator.Frustrated_Ladder(N, J1, J2)

startTime = time.time() 

Simulation = Simulator.DMQMC_Core(N, H, dbeta, init_pop)
Simulation.run_finite_beta(no_loops, final_beta, resolution)
Simulation.close_outputs()

endTime = time.time() 

print('Total time taken', (endTime-startTime)/60, 'minutes')




#%%

# Growth rate analysis
# Running a diagnostic tools
# ***Input parameters here***
# Number of spins
N = 8
# Coupling strength
J1 = 1
J2 = 1
# Finite difference beta step
dbeta = 1/8


# ***Select the Hamiltonian here***
H = H_generator.Frustrated_Ladder(N, J1, J2)

analyser = Analysis.Growth_Analyser(H, N, dbeta)

analyser.get_all_fittings()
analyser.rate_fitting()
analyser.run_diagnostics()

#%%

# Fractional Error analysis
# Running a diagnostic tools
# ***Input parameters here***
# Number of spins
N = 8
# Coupling strength
J1 = 1
J2 = 1
# Finite difference beta step
dbeta = 1/8

# ***Select the Hamiltonian here***
H = H_generator.Frustrated_Ladder(N, J1, J2)

analyser = FEA.Fractional_Error(N, H)
analyser.fGraph(100, dbeta)
