#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:59:38 2021

@author: darrenlean
"""
'''
This script helps to explore the database
 
NOT essential for any part of the DMQMC at all
'''

import os

def folder_finder(directory):
    folder_names = []
    for entry_name in os.listdir(directory):
        entry_path = os.path.join(directory, entry_name)
        if os.path.isdir(entry_path):
            folder_names.append(entry_name)
    return folder_names

def textfile_finder(directory):
    '''
    This is a helper function to navigate through the directory
    It returns the existing file names and number of files
    '''
    fileNames = []
    for file in os.listdir(directory+'/'):
        if file.endswith(".txt"):
            fileNames.append(os.path.join(file))
    return fileNames

def get_all_loop_indices(data_directory):
    '''
    This helper function retrieve all the loop indices
    '''
    folder_names = []
    for entry_name in os.listdir(data_directory):
        entry_path = os.path.join(data_directory, entry_name)
        if os.path.isdir(entry_path):
            folder_names.append(int(entry_name))
    
    return folder_names
  
def navigator():
    
    directory = 'Database/'
    print("Folders in " + directory + ":", folder_finder(directory))
    H = input("Select the Hamiltonian:     ")
    if H not in folder_finder(directory):
        print(H+" is not found")    
    else:
        directory += H + "/"
    
    print("")
    temp = []
    for k in folder_finder(directory):
        temp.append(k[2])
    print("Folders in " + directory + ":", temp)
    N = input("Select the number of spins:     ")
    if N not in temp:
        print(N+" is not found")    
    else:
        directory += "N=" + N + "/"
        
    N = int(N)
    
    print("")
    temp = []
    for k in folder_finder(directory):
        temp.append(k[6:])
    print("Folders in " + directory + ":", temp)
    dbeta = input("Select finite difference dbeta:     ")
    if dbeta not in temp:
        print(dbeta+" is not found")    
    else:
        directory += "dbeta=" + dbeta + "/"
    
    dbeta = float(dbeta)
    
    print("")
    temp = []
    for k in folder_finder(directory):
        temp.append(k[8:])
    print("Folders in " + directory + ":", temp)
    init_pop = input("Select initial diagonal population:     ")
    if init_pop not in temp:
        print(init_pop+" is not found")    
    else:
        directory += "initpop=" + init_pop
    
    init_pop = float(init_pop)
    
    all_loops = get_all_loop_indices(directory)
    
    print('Loops found: ', all_loops)
    
    print('Raw textfiles:', textfile_finder(directory+'/'+str(all_loops[0])+'/'))
