# %% Abstract
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:28:42 2021

@author: Leoanrdo Sito, leonardosito66@gmail.com

Importing data from txt file and cleaning
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

cal_data = pd.read_csv(cal_file, 
                 index_col = (0)) # Indexes are the dates

# Here I want to store all the intercept 
slope_scatter = []
intercept_scatter = []

for index in range(len(cal_data.columns)):
    if index % 3 == 0:
        slope_scatter.append(cal_data.iloc[0,index])
        intercept_scatter.append(cal_data.iloc[1,index])
    


# %% Importing data
# The firs number in the txt file, tells how many rows to jump before we have any
# actual infromation, so we get it as an int

# Folder where all the files are
folder = 'TesBeam_202110'
# Empty list to store dates
date = []
# Very bad practice
outputs = pd.DataFrame()

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
                         header = None)
        
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], dayfirst = True)
        df.iloc[:,0] = df.iloc[:,0].dt.strftime('%d %b, %H:%M')
        
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
        
        first_column = int(tot_num + sum(df.iloc[0,1:col_num_channel]*2)+1)
        
        # Let's make another dataframe with only the columns I want
        
        # First define the matrix of the right size
        array = np.zeros((df.shape[0], df.iloc[0,col_num_channel]))
        
        # Conserviamoci le date in una struttura esterna
        lista = df.iloc[:,0].values.tolist()
        for i in lista:
            date.append(i)
        # I would like to see all the data togheter
        
        # Filling the array
        for i in range(0,int(df.iloc[0,col_num_channel]/2)):
            # I will be taking one column each 4
            array[:,i] = (df.iloc[:,(first_column+4*i)]-intercept_scatter[i])/slope_scatter[i]
            
        # Now before exporting everything I should convert to temperatures using
        # the cal curve obtained
            
        data = pd.DataFrame(array)
        
        data.index = df.index
        outputs = outputs.append(data, ignore_index=True)    
        
        
#outputs.index = pd.to_datetime(outputs.index)        
#outputs = outputs.sort_index(axis=0, ascending=False)

# %% Plotting
# The first 8 are 4.1, the others 3.1 ??



for i in range(8):

    plt.figure() # Create new figure
    ax = plt.axes() # Create a new set of axes
    
    # Manually inserting indexes and labels
    tick_indx =  [0, 2219, 4440, 6660, 8879, 11100, 13317]
    ticks = [date[i] for i in tick_indx]
    
    plt.plot(outputs.index, outputs.iloc[:,i], label = f'4.1_{i+1}', linewidth=0.5)
    plt.xticks(tick_indx, ticks, rotation=-90, style='normal') # Rotate the ticks for readability
    
    # Get rid of scienfic notation only on y axes
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
    #plt.ylim(1527.4, 1527.5)
    
    plt.legend()
    
    plt.title('Temperature')
    plt.xlabel('Date')
    plt.ylabel(r'T ($^\circ C$)')
    
    plt.show()
