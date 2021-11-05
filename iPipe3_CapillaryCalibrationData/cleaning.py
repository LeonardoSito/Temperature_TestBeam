# %% Importing packages and modules

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import datetime # To convert Unix data
import matplotlib.dates as mdates # To change data formattin in plots

# To manage ticks in graphs, get rid of scientific notation
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

import statistics

# To smooth data
from scipy.signal import savgol_filter

# Defining a style to use in the whole document
plt.style.use('ggplot')


# data --> Data Frame containing the data
# column_data --> Specify the column of the data frame
# Delta --> Range of the derivative values you want to take

def clean(data, column_data, Delta, start_value, plateau_number, plot_derivative, plot_slots, results_dir):

    # Filtering data, comment to unfilter
    filtered = savgol_filter(data.iloc[:,column_data], 99, 3)
    # Derivative applied on filtered data
    der = np.diff(filtered)
    
    # der = np.diff(data.iloc[:,column_data])
    
    if plot_derivative == True:
        # Define matrix which will be filled with indexes
        # plateau_idx = np.zeros(shape = [len(data.index),100])
        
        # I want to look at the derivative of the data (with respect to time)
        plt.figure()
        ax = plt.axes()
        # When doing the derivative I lose 1 data point, so the x axis has to have one less
        plt.plot(data.index[1:], der, color='#374c80', label='Derivative', linewidth=0.1)
        plt.xticks(rotation=-90, style='normal') # Rotate the ticks for readability
    
        # Get rid of scienfic notation only on y axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
        plt.legend()
    
        plt.title('Derivative time')
        plt.xlabel('Date(dd/mm/yyyy)')
        plt.ylabel(r'Derivative')
    
        plt.savefig(results_dir + '\Derivative.png', dpi=600, bbox_inches = 'tight')

    # Defining index to fill matrix and parameters
    i = 0
    j = 0
    
    lista = [0]
    ranges = [0]
    
    for idx, val in enumerate(der):
        if idx > start_value:
            if (val < Delta and val > -Delta): # Da aggiungere una condizione sul valore
                lista.append(idx)    
            else:
                    max_val = np.max(lista)
                    lista[:] = [max_val if x==0 else x for x in lista]
                    min_val = np.min(lista)
                    ranges.append(min_val)
                    ranges.append(max_val)
                    lista = [0]
    
    # I fill one last time the matrix to consider the last element (i.e. the one
    # with no more peaks after it)                
    max_val = np.max(lista)
    lista[:] = [max_val if x==0 else x for x in lista]
    min_val = np.min(lista)
    ranges.append(min_val)
    ranges.append(max_val)                
                    
    # Removing zeros
    ranges = [i for i in ranges if i!=0]            
    ranges = np.array([ranges]) # Convert to np array
    ranges = ranges.transpose() # I want a column
    
    # ranges is going to be a matrix with the max and min index of each slot
    size = int(len(ranges)/2)
    ranges = ranges.reshape(size, 2)
    ranges = ranges.transpose()
    
    # %%%% Plotting the plateau that are going to be taken
    
    if plot_slots == True:
        plt.figure() # Create new figure
        ax1 = plt.axes() # Create a new set of axes
        
        plt.plot(data.index, data.iloc[:,column_data], label = "lambda", linewidth=0.5)
        plt.xticks(rotation=-90, style='normal') # Rotate the ticks for readability
        
        # Get rid of scienfic notation only on y axes
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
        ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        
        # Plot a line to see what I am taking
        for i in range(size):
            l = ranges[0,i]
            m = ranges[1,i]
            plt.axvspan(data.index[l], data.index[m], alpha=0.3)
        
        plt.legend()
        
        plt.title('What I am taking')
        plt.xlabel('Date(dd/mm/yyyy)')
        plt.ylabel(r'$\lambda$ or T')
        
        plt.savefig(results_dir + f'\FilteredData{column_data}.png', dpi=600, bbox_inches = 'tight')
        
    # %%%% Extracting the slots and means, variances
    
    # Taking only the acceptable values
    filtered_data = []
    
    for i in range(size):
        l = ranges[0,i]
        m = ranges[1,i]
        for j in range(l, m, 1):
            filtered_data.append(data.iloc[j,column_data]) 
    
    # Sorting the list
    filtered_data.sort()
    
    # Defining intervals for the occurrances of each value
    max_lambda = np.max(data.iloc[:,column_data]) # nm
    min_lambda = np.min(data.iloc[:,column_data]) # nm
    
    Delta_lambda_tot = max_lambda - min_lambda # nm
        
    step = Delta_lambda_tot / (plateau_number-1) 
    
    # Defining a list of all the mid points between two plateaus
    upper_limit_step = []
    top_step = min_lambda + step/2
    
    for i in range(plateau_number):
        upper_limit_step.append(top_step + i*step)
        
    N = [] # Number of elements summed
    numb = np.zeros(11) # Number of elements summed
    mu = np.zeros(11) # Mean of those elements
    std = np.zeros(11)# Variance of those elements
    
    for indx, i in enumerate(upper_limit_step):
        for element in filtered_data:
            if element < i and element > i-step:
                N.append(element)    
        mu[indx] = np.mean(N)
        std[indx] = np.std(N, ddof=1) # To divide by (N-1)   
        numb[indx] = len(N)  
        N = []
    
    return mu, std, numb
    