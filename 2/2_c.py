import numpy as np
import pandas as pd
import random

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

file_path = '/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/hmeq.csv'
data = pd.read_csv(file_path)

data['BAD'].replace(np.nan, 99, inplace=True)
data['REASON'].replace(np.nan, 'MISSING', inplace=True)

data['CaseID'] = data.index.values
data_wIndex = data.set_index("CaseID")

sampleData = data_wIndex[['BAD','LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']].reset_index()

random.seed(a = 20230101)

trainIndex = []
testIndex = []

for catb in [0,1]:
   for catr in ['HomeImp','DebtCon','MISSING']:
      obsIndex = sampleData[(sampleData['BAD'] == catb) & (sampleData['REASON'] == catr)].index
      trIndex, ttIndex = simpleRandomSample (obsIndex, trainFraction = 0.7)
      trainIndex.extend(trIndex)
      testIndex.extend(ttIndex)

print("Number of Observations in the Train Partition : " + str(len(trainIndex)))
print("Number of Observations in the Test Partition : " + str(len(testIndex)))
print()

trainData = sampleData.loc[trainIndex][['BAD','LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
testData = sampleData.loc[testIndex][['BAD','LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]

print("Frequency distributions of BAD in Train partition : ")
print(trainData['BAD'].value_counts(dropna=False))
print("Frequency distributions of BAD in Test partition : ")
print(testData['BAD'].value_counts(dropna=False))
print()

print("Mean and Standard Deviations in the Train Data")

print("DEBTINC")
print("Mean : " + str(trainData['DEBTINC'].mean()))
print("Standard Deviation : " + str(trainData['DEBTINC'].std()))

print("LOAN")
print("Mean : " + str(trainData['LOAN'].mean()))
print("Standard Deviation : " + str(trainData['LOAN'].std()))

print("MORTDUE")
print("Mean : " + str(trainData['MORTDUE'].mean()))
print("Standard Deviation : " + str(trainData['MORTDUE'].std()))

print("VALUE")
print("Mean : " + str(trainData['VALUE'].mean()))
print("Standard Deviation : " + str(trainData['VALUE'].std()))
print()

print("Mean and Standard Deviations in the Test Data")
print("DEBTINC")
print("Mean : " + str(testData['DEBTINC'].mean()))
print("Standard Deviation : " + str(testData['DEBTINC'].std()))

print("LOAN")
print("Mean : " + str(testData['LOAN'].mean()))
print("Standard Deviation : " + str(testData['LOAN'].std()))

print("MORTDUE")
print("Mean : " + str(testData['MORTDUE'].mean()))
print("Standard Deviation : " + str(testData['MORTDUE'].std()))

print("VALUE")
print("Mean : " + str(testData['VALUE'].mean()))
print("Standard Deviation : " + str(testData['VALUE'].std()))