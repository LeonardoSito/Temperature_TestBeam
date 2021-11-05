# %% Technical Info
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:46:30 2021

@author: leona
"""

# %% Importing packages and modules

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import datetime # To convert Unix data

# To manage ticks in graphs, get rid of scientific notation
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

import statistics

# Defining a style to use in the whole document
plt.style.use('ggplot')

# %% Importing data and cleaning
data = pd.read_csv("iPipe3_CapillaryCalibrationData/EndCap_31-41_Sync/EndCap_41_2.csv", index_col='UnixTime_FBG')

# Time data are in Unix (seconds from 1st Jan 1970). 
# There is a time difference beetwen the data acquired from the FBG and PT1000
# We see that this time difference may not be significant
max_time_diff = np.max(data.iloc[:,2]) #seconds
# Time step FBG first two samples
time_step11 = data.index[3]-data.index[2] 
# Time step PT1000 first two samples
time_step21 = data.iloc[4,1]-data.iloc[3,1]

# We change the date from Unix to GMT +2

data.index = pd.to_datetime(data.index, unit = 's')
# Change how date is displayed
# data.index = data.index.strftime('%h %d %m %Y')
data.iloc[:,1] = pd.to_datetime(data.iloc[:,1], unit = 's')

# %% Plotting the data - time distribution of lambda

fig_1 = plt.figure() # Create new figure
ax1 = plt.axes() # Create a new set of axes

plt.plot(data.index, data.iloc[:,0], label = "lambda", linewidth=0.5)
plt.xticks(rotation=-90, style='normal') # Rotate the ticks for readability

# Get rid of scienfic notation only on y axes
# ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

plt.legend()

plt.title('Lambda')
plt.xlabel('Date(dd/mm/yyyy)')
plt.ylabel(r'$\lambda$ (nm)')

fig_1.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig('TimeData1.png', dpi=600)

# %% Plotting the data togheter
fig_3, ax1 = plt.subplots()

ax1.set_xlabel('date')
ax1.set_ylabel(r'$\lambda$ (nm)')
ax1.plot(data.index, data.iloc[:,0], label = r'$\lambda$', linewidth=0.5)
ax1.tick_params(axis='y')
plt.xticks(rotation=-90, style='normal')

# To get rid of the scientific notation
ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel(r'$T$')  # we already handled the x-label with ax1
ax2.plot(data.index, data.iloc[:,3], label = "T", linewidth=0.5)
ax2.tick_params(axis='y')

fig_3.tight_layout()  # otherwise the right y-label is slightly clipped

plt.legend()

plt.title('Lambda and Temperature')
plt.savefig('TimeData2.png', dpi=600)
plt.show()


# %% Scattering the data - T vs lambda
fig_4 = plt.figure()
ax4 = plt.axes() # Create a new set of axes

fig_4 = plt.scatter(data.iloc[:,3], data.iloc[:,0], s = 0.75, marker = '.', label = "Scatter")

# %% Simple linear regression
b, a = np.polyfit(data.iloc[:,3], data.iloc[:,0], 1)

plt.plot(data.iloc[:,3], (a + b*data.iloc[:,3]), 
         label = "Linear fit", color = '#ffa600', 
         linewidth=0.75)

plt.legend()

# Get rid of scientific notation
# ax4.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax4.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))


plt.title('Calibration Curve')
plt.xlabel(r'T ($^\circ$C)')
plt.ylabel(r'$\lambda$ (nm)')

plt.savefig('Scatter_Fit.png', dpi=600, bbox_inches = 'tight')

# %% More in depth

# I want to look at the derivative of the data (with respect to time)


fig_5 = plt.figure()
ax5 = plt.axes()
plt.plot(data.index[1:], np.diff(data.iloc[:,0]), color='#374c80', label='Derivative', linewidth=0.1)
plt.xticks(rotation=-90, style='normal') # Rotate the ticks for readability

# Get rid of scienfic notation only on y axes
# ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

plt.legend()

plt.title('Derivative of lambda in time')
plt.xlabel('Date(dd/mm/yyyy)')
plt.ylabel(r'Derivative (m/s)')

plt.savefig('Derivative.png', dpi=600, bbox_inches = 'tight')

column_data = 0

der = np.diff(data.iloc[:,column_data])
lam = data.to_numpy(dtype=None, copy=False)[:,0]

# Define matrix which will be filled with indexes
# plateau_idx = np.zeros(shape = [len(data.index),100])

# Defining index to fill matrix and paramiters
i = 0
j = 0
Delta = 0.0001 # Max displacement
DeltaUp = 0.01
DeltaDown = -0.005
start_value = 10 # Discard the first few elements
# Algorithm to fill the matrix
# for idx, val in enumerate(der):

#     if val < Delta and val > -Delta:
#         plateau_idx[i ,j] = idx
#         i = i+1
#     if i != 0:    
#         if val < DeltaDown or val > DeltaUp:
#             j = j+1
#             i = 0

lista = [0]
ranges = [0]

for idx, val in enumerate(der):
    if idx > start_value:
        if (val < Delta and val > -Delta): # Da aggiungere una condizione sul valore
            lista.append(idx)    
        if val < DeltaDown or val > DeltaUp: # else (metti questo)
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
# plateau_idx.transpose()
# plateau_idx = plateau_idx[plateau_idx!=0]
fig_6 = plt.figure() # Create new figure
ax1 = plt.axes() # Create a new set of axes

plt.plot(data.index, data.iloc[:,column_data], label = "lambda", linewidth=0.5)
plt.xticks(rotation=-90, style='normal') # Rotate the ticks for readability

# Get rid of scienfic notation only on y axes
# ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

# Plot a line to see what I am taking
for i in range(size):
    l = ranges[0,i]
    m = ranges[1,i]
    plt.axvspan(data.index[l], data.index[m], alpha=0.3)

plt.legend()

plt.title('Lambda')
plt.xlabel('Date(dd/mm/yyyy)')
plt.ylabel(r'$\lambda$ (nm)')

plt.savefig('TimeData4.png', dpi=600, bbox_inches = 'tight')

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

plateau_number = 11

step = Delta_lambda_tot / (plateau_number-1) 

# Defining a list of all the mid points between two plateaus
upper_limit_step = []
top_step = min_lambda + step/2

for i in range(plateau_number):
    upper_limit_step.append(top_step + i*step)
    
N = [] 
numb = np.zeros(11) # Number of elements summed
mu = np.zeros(11) # Mean of those elements
std = np.zeros(11)# Variance of those elements

for indx, i in enumerate(upper_limit_step):
    for element in filtered_data:
        if element < i and element > i-step:
            N.append(element)    
    
    mu[indx] = np.mean(N)
    std[indx] = np.std(N)
    numb[indx] = len(N)    
    N = []

# L'array mu e std sono le uscite che mi servono

import cleaning as cl

mu_2, std_2, numb_2 = cl.clean(data, 3, 0.09)

fig_7 = plt.figure()

ax7 = plt.axes()

#plt.scatter(mu_2, mu, s = 0.75, marker = '.')

errorx = std_2
errory = std
# La fascia di errore andrebbe divisa per la radice del numero di campioni
# errorx = std_2/np.sqrt(numb)
# errory = std/np.sqrt(numb)
plt.errorbar(mu_2, mu, xerr = errorx, yerr = errory, fmt='none', elinewidth = 0.5, label = 'Error bars')

b1, a1 = np.polyfit(mu_2, mu, 1)

plt.plot(mu_2, (a1 + b1*mu_2), 
          label = "Linear fit", color = '#ffa600', 
          linewidth=0.5)

b, a = np.polyfit(data.iloc[:,3], data.iloc[:,0], 1)

plt.plot(data.iloc[:,3], (a + b*data.iloc[:,3]), 
         label = "Linear fit (old)", color = '#003f5c', 
         linewidth=0.2)


plt.legend()

# Get rid of scientific notation
# ax4.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax7.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))



plt.title('Calibration Curve')
plt.xlabel(r'T ($^\circ$C)')
plt.ylabel(r'$\lambda$ (nm)')

plt.savefig('Final.png', dpi=600, bbox_inches = 'tight')