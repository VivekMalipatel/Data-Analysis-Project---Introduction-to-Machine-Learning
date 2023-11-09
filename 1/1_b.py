import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def shimazaki_criterion (x, d_list):

    number_bins = []
    matrix_boundary = []
    criterion_list = []

    x_min = np.min(x)
    x_max = np.max(x)
    x_mean = np.mean(x)

    # Loop through the bin width candidates
    for delta in d_list:
         x_middle = delta * np.round(x_mean / delta)
         n_bin_left = np.ceil((x_middle - x_min) / delta)
         n_bin_right = np.ceil((x_max - x_middle) / delta)
         x_low = x_middle - n_bin_left * delta

         # Assign observations to bins starting from 0
         list_boundary = []
         n_bin = n_bin_left + n_bin_right
         bin_index = 0
         bin_boundary = x_low
         for i in np.arange(n_bin):
            bin_boundary = bin_boundary + delta
            bin_index = np.where(x > bin_boundary, i+1, bin_index)
            list_boundary.append(bin_boundary)

         # Count the number of observations in each bins
         uvalue, ucount = np.unique(bin_index, return_counts = True)

         # Calculate the average frequency
         mean_ucount = np.mean(ucount)
         ssd_ucount = np.mean(np.power((ucount - mean_ucount), 2))
         criterion = (2.0 * mean_ucount - ssd_ucount) / delta / delta

         number_bins.append(n_bin)
         matrix_boundary.append(list_boundary)
         criterion_list.append(criterion)
        
    return(number_bins, matrix_boundary, criterion_list)

file_path = '/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/Gamma4804.csv'

data = pd.read_csv(file_path)
X = data['x']

d_list = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]
number_bins, matrix_boundary, criterion_list = shimazaki_criterion (X, d_list)

for delta, bin_boundary, criterion, n in zip(d_list, matrix_boundary, criterion_list, number_bins):
   plt.figure(figsize = (10,6), dpi = 200)
   plt.hist(X, bins = bin_boundary, align = 'mid')
   plt.title('Delta = ' + str(delta) + '  C(d) = ' + str(criterion) + "  Number of bins = " + str(n))
   plt.ylabel('Number of Observations')
   plt.grid(axis = 'y')
   plt.show() 