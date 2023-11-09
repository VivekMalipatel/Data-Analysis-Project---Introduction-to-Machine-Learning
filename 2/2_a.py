import pandas as pd

file_path = '/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/hmeq.csv'

data = pd.read_csv(file_path)

print("Number of Observations : " + str(len(data)))
print()
print("Frequency distributions of BAD : ")
print(data['BAD'].value_counts(dropna=False))
print("Missing Values : " + str(data['BAD'].isna().sum()))
print()
print("DEBTINC")
print("Mean : " + str(data['DEBTINC'].mean()))
print("Standard Deviation : " + str(data['DEBTINC'].std()))
print()
print("LOAN")
print("Mean : " + str(data['LOAN'].mean()))
print("Standard Deviation : " + str(data['LOAN'].std()))
print()
print("MORTDUE")
print("Mean : " + str(data['MORTDUE'].mean()))
print("Standard Deviation : " + str(data['MORTDUE'].std()))
print()
print("VALUE")
print("Mean : " + str(data['VALUE'].mean()))
print("Standard Deviation : " + str(data['VALUE'].std()))