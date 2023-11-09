import numpy as np
import pandas as pd
import random

def simpleRandomSample (obsIndex, trainFraction = 0.8):

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

file_path = '/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/Fraud.csv'
data = pd.read_csv(file_path)

random.seed(a = 20230225)

trainIndex = []
testIndex = []

data_wIndex = data.set_index("CASE_ID")

sampleData = data_wIndex[['FRAUD','TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']].dropna().reset_index()

for cat in [0,1]:
   obsIndex = sampleData[sampleData['FRAUD'] == cat].index
   trIndex, ttIndex = simpleRandomSample (obsIndex, trainFraction = 0.8)
   trainIndex.extend(trIndex)
   testIndex.extend(ttIndex)

print("Number of Observations in the Train Partition : " + str(len(trainIndex)))
print("Number of Observations in the Test Partition : " + str(len(testIndex)))
