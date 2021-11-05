# %% Importing packages and modules

import os as os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import datetime # To convert Unix data
import matplotlib.dates as mdates # To change data formattin in plots

# To smooth data
from scipy.signal import savgol_filter

# To manage ticks in graphs, get rid of scientific notation
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

# To have another fitting function
import statistics
from scipy import stats

# Another kind of fit
from scipy.odr import *

# We need a function to filter the data which is in the same folder
import cleaning as cl

# Autocorrelation function
from scipy import signal
#import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf

# Defining a style to use in the whole document
plt.style.use('ggplot')

# %% What I need to declare for output
indexes_out = ['Slope', 'Intercept', 'Slope error', 'Intercept error']
outputs = pd.DataFrame()

# Vector going from 1 to 8 linspaced
file_names = np.linspace(1,8,8)

# %% Importing data and cleaning

for file_num in file_names:
    
    # Casting to int to avoid issues
    file_num = int(file_num)
    
    # File paths
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, f'EndCap_41_{file_num}')
    
    # Create directory if non existant
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    data = pd.read_csv(f"iPipe3_CapillaryCalibrationData/EndCap_31-41_Sync/EndCap_41_{file_num}.csv", index_col='UnixTime_FBG')
    
    # Time data are in Unix (seconds from 1st Jan 1970). 
    # There is a time difference beetwen the data acquired from the FBG and PT1000
    
    # We see that this time difference may not be significant
    # max_time_diff = np.max(data.iloc[:,2]) #seconds
    # Time step FBG first two samples
    # time_step11 = data.index[3]-data.index[2] 
    # Time step PT1000 first two samples
    # time_step21 = data.iloc[4,1]-data.iloc[3,1]
    
    # We change the date from Unix to GMT +2
    
    data.index = pd.to_datetime(data.index, unit = 's')
    
    # If I want to visualize only from day of the month to seconds
    # data.index = data.index.strftime('%d %b, %H:%M %S')
    
    # Change how date is displayed
    # data.index = data.index.strftime('%h %d %m %Y')
    data.iloc[:,1] = pd.to_datetime(data.iloc[:,1], unit = 's')
    
    # %% Lambda VS Time
    
    plt.figure() # Create new figure
    ax1 = plt.axes() # Create a new set of axes
    
    plt.plot(data.index, data.iloc[:,0], label = r'$lambda$', linewidth=0.5)
    plt.xticks(rotation=-90, style='normal') # Rotate the ticks for readability
    
    # Get rid of scienfic notation only on y axes
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
    plt.legend()
    
    plt.title('Lambda')
    plt.xlabel('Date')
    plt.ylabel(r'$\lambda$ (nm)')
    
    # Uncomment to save figure
    plt.savefig(results_dir + '\LambdaVSTime.png', dpi=600, bbox_inches = 'tight')
    
    # %% Lambda VS Time & Temperature VS Time
    fig_3, ax1 = plt.subplots()
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel(r'$\lambda$ (nm)')
    ax1.plot(data.index, data.iloc[:,0], label = r'$\lambda$', linewidth=0.5, color = '#f95d6a')
    ax1.tick_params(axis='y')
    plt.xticks(rotation=-90, style='normal')
    
    # To get rid of the scientific notation
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    ax2.set_ylabel(r'$T$')  # we already handled the x-label with ax1
    ax2.plot(data.index, data.iloc[:,3], label = "T", linewidth=0.5, color = '#ffa600')
    ax2.tick_params(axis='y')
    
    fig_3.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.legend()
    
    plt.title('Lambda and Temperature')
    plt.savefig(results_dir + '\TemperatureVSTime', dpi=600, bbox_inches = 'tight')
    plt.show()
    
    
    # %% Scattering the data - T vs lambda
    plt.figure()
    ax4 = plt.axes() # Create a new set of axes
    
    plt.scatter(data.iloc[:,3], data.iloc[:,0], s = 0.75, marker = '.', label = "Scatter")
    
    # %%%% 0. Linear Fit
    
    fit_data_0 = np.polyfit(data.iloc[:,3], data.iloc[:,0], 1, cov = True)
    line_0 = fit_data_0[0]
    cov_matrix_0 = fit_data_0[1]
    
    plt.plot(data.iloc[:,3], (line_0[1] + line_0[0]*data.iloc[:,3]), 
             label = "Linear fit", color = '#ffa600', 
             linewidth=0.75)
    
    plt.legend()
    
    # Get rid of scientific notation
    # ax4.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax4.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
    
    plt.title('Calibration Curve from scattered data')
    plt.xlabel(r'T ($^\circ$C)')
    plt.ylabel(r'$\lambda$ (nm)')
    
    plt.savefig(results_dir + '\CalCurve_0.png', dpi=600, bbox_inches = 'tight')
    
    # %% More advanced
    # We have already imported: import cleaning as cl
    
    # To see what parameter to pass look at the module
    # delta for lambda like 0.001
    # delta for T like 0.09
    mu_lambda, std_lambda, numb_lambda = cl.clean(data, 0, 0.0001, 10, 11, True, True, results_dir)
    mu_T, std_T, numb_T = cl.clean(data, 3, 0.009, 0, 11, True, True, results_dir)
    
    # Standard error
    errorx = std_T#/np.sqrt(numb_T)
    errory = std_lambda#/np.sqrt(numb_lambda)
    
    # %%%% Plot means and error bars
    plt.figure()
    ax = plt.axes()
    
    plt.errorbar(mu_T, mu_lambda, xerr = errorx, yerr = errory, fmt='none', elinewidth = 0.5, label = 'Error bars')
    
    
    # %%%% 1. Linear fit of means from numpy
    
    fit_data_1 = np.polyfit(mu_T, mu_lambda, 1, cov = True)
    line_1 = fit_data_1[0]
    cov_matrix_1 = fit_data_1[1]
    
    # %%%% 2. Linear fit of means from statistics - unused
    # b1, a1, r_value, p_value, std_err = stats.linregress(mu_T, mu_lambda)
    
    # %%%% 3. Linear fit weighed - Deming regression
        # https://docs.scipy.org/doc/scipy/reference/odr.html
        # P. T. Boggs and J. E. Rogers, “Orthogonal Distance Regression,” in “Statistical analysis of measurement error models and applications: proceedings of the AMS-IMS-SIAM joint summer research conference held June 10-16, 1989,” Contemporary Mathematics, vol. 112, pg. 186, 1990.
        # Define a function (quadratic in our case) to fit the data with.
        # https://towardsdatascience.com/error-in-variables-models-deming-regression-11ca93b6e138
    def f(B, x):
        '''Linear function y = m*x + b'''
        # B is a vector of the parameters.
        # x is an array of the current x values.
        # x is in the same format as the x passed to Data or RealData.
        #
        # Return an array in the same format as y passed to Data or RealData.
        return B[0]*x + B[1]
    
    # Create a model for fitting.
    linear = Model(f)
    
    # Create a RealData object using our initiated data from above.
    data_fit = RealData(mu_T, mu_lambda, sx=errorx, sy=errory)
    
    # Set up ODR with the model and data.
    myodr = ODR(data_fit, linear, beta0=[1., 2.])
    
    # Run the regression.
    myoutput = myodr.run()
    # myoutput.pprint()
    
    # Std errors of the two fitting parameters
    intercept_std_err = myoutput.sd_beta[1]
    slope_std_err = myoutput.sd_beta[0]
    
    y_fit = f(myoutput.beta, mu_T)
    
    # %%%% Plot ODR
    plt.plot(mu_T, y_fit, 
              label = "Linear fit_weighted", color = '#f95d6a', 
              linewidth=0.5)
    
    # %%%% Plot numpy
    plt.plot(mu_T, (line_1[1] + line_1[0]*mu_T), 
              label = "Linear fit", color = '#003f5c', 
              linewidth=0.5)
    
    # %%%% Third order fit 
    coeff_third = np.polyfit(mu_T, mu_lambda, 3)
    
    # %%%% Plot third order fit
    plt.plot(mu_T, (coeff_third[0]*(mu_T)**3 + coeff_third[1]*(mu_T)**2 + coeff_third[2]*mu_T + coeff_third[3]), 
              label = "Thrid order fit", color = '#a05195', 
              linewidth=0.5)
    
    
    # %%%% Plot over the previous result
    plt.plot(data.iloc[:,3], (line_0[1] + line_0[0]*data.iloc[:,3]), 
             label = "Linear fit old", color = '#ffa600', 
             linewidth=0.5)
    
    plt.legend()
    
    # Get rid of scientific notation
    # ax4.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
    
    plt.title('Calibration Curve')
    plt.xlabel(r'T ($^\circ$C)')
    plt.ylabel(r'$\lambda$ (nm)')
    
    plt.savefig(results_dir + '\CalCurve_1.png', dpi=600, bbox_inches = 'tight')
    
    # %% Outputs [Slope, Intercept, Slope error, Intercept error]
    
    dictionary = {
        #"outputs": ['Slope', 'Intercept', 'Slope error', 'Intercept error'],
        f"scatter_results_{file_num}": [line_0[0], line_0[1], np.sqrt(cov_matrix_0[0,0]), np.sqrt(cov_matrix_0[1,1])],
        f"filtered_results_{file_num}": [line_1[0], line_1[1], np.sqrt(cov_matrix_1[0,0]), np.sqrt(cov_matrix_1[1,1])],
        f"filtered_results_ODR_{file_num}": [myoutput.beta[0], myoutput.beta[1], slope_std_err, intercept_std_err]   
    }
    
    DataFrame = pd.DataFrame(dictionary)
    
    outputs = pd.concat([outputs, DataFrame], axis=1)

# %% Refreshing the dataframe and exporting as csv

outputs.index = indexes_out
outputs.to_csv('output_file.csv')

# Sistemare bande di errore e deviazione standard