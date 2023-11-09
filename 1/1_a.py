import pandas as pd
import numpy as np

file_path = '/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/Gamma4804.csv'

data = pd.read_csv(file_path).values.tolist()

print("Count: ", round(len(data),7))
print("Mean: ", round(np.mean(data),7))
print("Standard Deviation: ", round(np.std(data),7))
print("Minimum: ", round(np.min(data),7))
print("25th Percentile: ", round(np.percentile(data, 25),7))
print("Median: ", round(np.median(data),7))
print("75th Percentile: ", round(np.percentile(data, 75),7))
print("Maximum: ", round(np.max(data),7))