import pandas as pd

file_path = '/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/Fraud.csv'
data = pd.read_csv(file_path)

fraud_count = (data['FRAUD']==1).sum()
per_fraud = round((fraud_count/len(data)*100),4)

print("Percent of investigations are found to be frauds : " + str(per_fraud))