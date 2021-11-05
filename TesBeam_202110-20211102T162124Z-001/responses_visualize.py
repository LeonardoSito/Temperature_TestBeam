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

# %% Importing data
# The first number in the txt file, tells how many rows to jump before we have any
# actual infromation, so we get it as an int

# Folder where all the files are
folder = 'TesBeam_202110'

# Getting a list with the name of all the files
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f.extend(filenames)
    break

for index, value in enumerate(f):
    if 'Responses' in value:
        file_name = f'{folder}\{value}'
        
        # Importing the data one by one
        with open(file_name) as file:
            jmp_num = int(file.readline())
        
        df = pd.read_csv(file_name, 
                          skiprows = jmp_num+1, # Jump the first rows
                          sep = '\s+', # Delimiter is a tab
                          skip_blank_lines = True,
                          header=None) 
        
        # %% Defining x vector
        
        Start =  1510 # Start value
        Delta = 0.005 # Step
        Points = 16001 # Number of points
        
        x = np.linspace(Start, Start+Delta*(Points-1),Points)
        
        wavelenghts = []
        for i in range(Points):
            wavelenghts.append(Start+Delta*i)
        
        # %% Assigning channel names
        # Empty list that will contain the names
        names = []
        
        # Algorithm to compute the channel names
        for i in range(len(df)):
            first = int(np.floor(i/4)+1)
            second = int(i % 4 + 1)
            names.append(f'CH{first}.{second}')
        
        df.index = names
        
        # # %% Plotting one at the time
        
        # for i in range(len(df)):
        #     plt.figure()
        #     ax = plt.axes() # Create a new set of axes
            
        #     plt.plot(x, df.iloc[i,:], 
        #              label = df.index[i], 
        #              linewidth=0.75)
            
        #     plt.ylim(-65, -15)
            
        #     plt.legend()
            
        #     # Get rid of scientific notation
        #     ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            
        #     plt.title('Spectrum of ' + df.index[i])
        #     plt.xlabel(r'$\lambda$ (nm)')
        #     plt.ylabel(r'$dB$')
            
        # %% Plotting all together
        
        plt.figure()
        ax = plt.axes() # Create a new set of axes
        
        for i in range(len(df)):
            
            plt.plot(x, df.iloc[i,:], 
                     label = df.index[i], 
                     linewidth=0.75)
            
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            
        plt.ylim(-65, -15)
        # Get rid of scientific notation
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        
        plt.title(file_name)
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$dB$')






