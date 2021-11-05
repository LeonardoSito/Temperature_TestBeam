# %% Abstract
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:28:42 2021

@author: Leoanrdo Sito, leonardosito66@gmail.com

Importing data from txt file and cleaning
Plotting temperatures
"""

# %% Importing modules

# To see file names
from os import walk

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import datetime # To convert Unix data
import matplotlib.dates as mdates # To change data formattin in plots

# To manage ticks in graphs, get rid of scientific notation
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

# Defining a style to use in the whole document
plt.style.use('ggplot')

# %% Importing the csv with the cal curves - at the moment we are doing it only for
# endcap minus 
cal_file = 'output_file.csv'

# Setting this index we can choose the cal parameter we want to use
reg_type = 2

# reg_type = 0 --> ordinary linear regression on scattered data
# reg_type = 1 --> ordinary linear regression on filtered data
# reg_type = 2 --> ODR on filtered data

# We need to have an index to select the right column
col_index = 3 - reg_type
if reg_type == 0:
    col_index = 0

cal_data = pd.read_csv(cal_file, 
                 index_col = reg_type)

# Here I want to store all the intercept 
slope_scatter = []
intercept_scatter = []
slope_var_scatter = []
intercept_var_scatter = []

for index in range(len(cal_data.columns)):
    if index % 3 == col_index:
        slope_scatter.append(cal_data.iloc[0,index])
        intercept_scatter.append(cal_data.iloc[1,index])
        slope_var_scatter.append(cal_data.iloc[2,index])
        intercept_var_scatter.append(cal_data.iloc[3,index])

# %% Importing data
# The first number in the txt file, tells how many rows to jump before we have any
# actual infromation, so we get it as an int

# Folder where all the files are
folder = 'TesBeam_202110'
# Very bad practice
Temperatures = pd.DataFrame()
Errors = pd.DataFrame()

# Getting a list with the name of all the files
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f.extend(filenames)
    break

for index, value in enumerate(f):
    if 'Peak' in value:
        file_name = f'{folder}\{value}'
        
        # Importing the data one by one
        with open(file_name) as file:
            jmp_num = int(file.readline())
        
        df = pd.read_csv(file_name, 
                         skiprows = jmp_num, # Jump the first rows
                         delimiter='\t', # Delimiter is a tab
                         skip_blank_lines = True,
                         index_col = (0), # Indexes are the dates
                         header=None) 
        
        df.index = pd.to_datetime(df.index, dayfirst=True)
        
        # Number of total channels
        n_channels = 4
        n_sub_channels = 4
        tot_num = n_channels * n_sub_channels
        
        # Endcap minus is chanel 2.1
        channel = 2
        sub_channel = 1
        
        # Remember that the index start form 0
        # To obtain the column number I have to multply chanel by 4 and then add sub_channel
        col_num_channel = int((channel-1) * n_sub_channels + sub_channel)
        
        first_column = int(tot_num + sum(df.iloc[0,0:col_num_channel-1]*2))
        last_column = int(first_column + df.iloc[0,col_num_channel-1]*2)
        
        # Let's make another dataframe with only the columns I want
        
        # First define the matrix of the right size
        array = np.zeros((df.shape[0], df.iloc[0,col_num_channel-1]))
        sigma_T = np.zeros((df.shape[0], df.iloc[0,col_num_channel-1]))
        # I would like to see all the data togheter
        
        # Filling the array
        for i in range(0,int(df.iloc[0,col_num_channel-1]/2)):
            # I will be taking one column each 4
            # This is the array of temperatures
            array[:,i] = (df.iloc[:,(first_column+4*i)]-intercept_scatter[i])/slope_scatter[i]
            
            # I need to have the relative uncertainty on each point
            sigma_T[:,i] = array[:,i]*(slope_var_scatter[i]/slope_scatter[i] + 
                                       intercept_var_scatter[i]/intercept_scatter[i])
                        
            
        data = pd.DataFrame(array)
        data.index = df.index
        
        sigma = pd.DataFrame(sigma_T)
        sigma.index = df.index
        
        Temperatures = Temperatures.append(data)
        Errors = Errors.append(sigma)

# outputs.index = pd.to_datetime(outputs.index)        
# outputs = outputs.sort_index(axis=0, ascending=False)

# %% Plotting
# The first 8 are 3.1, the others 4.1

for i in range(8):

    plt.figure() # Create new figure
    ax = plt.axes() # Create a new set of axes
    
    
    plt.plot(Temperatures.index, Temperatures.iloc[:,i], label = f'4.1_{i+1}', linewidth=0.5)
    plt.xticks(rotation=-90, style='normal') # Rotate the ticks for readability
    
    plt.fill_between(Temperatures.index, 
                     Temperatures.iloc[:,i]-3*Errors.iloc[:,i], 
                     Temperatures.iloc[:,i]+3*Errors.iloc[:,i],
                     color='gray', alpha=0.2)
    
    # Get rid of scienfic notation only on y axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
    plt.ylim(5, 15)
    
    plt.legend()
    
    titles = ['Calibration from scattered data',
              'Calibration from filtered data',
              'Calibration with ODR']
    
    plt.title(f'Temperature, {titles[reg_type]}')
    plt.xlabel('Date')
    plt.ylabel(r'T ($^\circ C$)')
    
    plt.show()
