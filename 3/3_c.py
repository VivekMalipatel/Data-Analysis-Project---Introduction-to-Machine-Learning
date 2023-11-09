import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier

file_path = '/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/Fraud.csv'
data = pd.read_csv(file_path)
fraud_count = (data['FRAUD']==1).sum()
EFR = round((fraud_count/len(data)),4)

def simpleRandomSample (obsIndex, trainFraction = 0.7):
   trainIndex = []

   nPopulation = len(obsIndex)
   nSample = np.round(trainFraction * nPopulation)
   kObs = 0
   iSample = 0
   for oi in obsIndex:
      kObs = kObs + 1
      U = random.random()
      uThreshold = (nSample - iSample) / (nPopulation - kObs + 1)
      if (U < uThreshold):
         trainIndex.append(oi)
         iSample = iSample + 1

      if (iSample == nSample):
         break

   testIndex = list(set(obsIndex) - set(trainIndex))
   return (trainIndex, testIndex)

def nbrs_metric (class_prob, y_class) :
   mce = np.mean(np.where(class_prob == y_class, 0, 1))
   return mce

data_wIndex = data.set_index("CASE_ID")

sampleData = data_wIndex[['FRAUD','TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']].dropna().reset_index()

random.seed(a = 20230225)

trainIndex = []
testIndex = []

obsIndex = sampleData.index
trainIndex, testIndex = simpleRandomSample (obsIndex, trainFraction = 0.8)

X_train = sampleData.loc[trainIndex][['TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']]
y_train = sampleData.loc[trainIndex]['FRAUD']

X_test = sampleData.loc[testIndex][['TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']]
y_test = sampleData.loc[testIndex]['FRAUD']

neigh_choice = range(2,8)

result = []

for k in neigh_choice:
    neigh = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
    nbrs = neigh.fit(X_train, y_train)

    cprob_train = nbrs.predict(X_train)
    cprob_traina = [1 if cprob_train[i] >= EFR else 0 for i in range(len(cprob_train))]
    MCE_Train = nbrs_metric (cprob_traina, y_train)

    cprob_test = nbrs.predict(X_test)
    cprob_testa = [1 if cprob_test[i] >= EFR else 0 for i in range(len(cprob_test))]
    MCE_Test = nbrs_metric (cprob_testa, y_test)
    
    result.append([k, MCE_Train, MCE_Test])

result_df = pd.DataFrame(result, columns = ['k', 'MCE_Train', 'MCE_Test'])
print(result_df)
