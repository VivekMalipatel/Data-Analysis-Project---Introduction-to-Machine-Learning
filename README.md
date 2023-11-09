# Data-Analysis-Project---Introduction-to-Machine-Learning
CS 484: Introduction to Machine Learning Assignment

# Data Analysis Project

This project consists of a series of data analysis tasks performed on various datasets. Below are the descriptions and questions that guide the analysis process.

## Question 1: Gamma4804.csv Analysis (30 points)

### Part (a) - Descriptive Statistics (10 points)

- Use the field `x` in the `Gamma4804.csv` file.
- Determine the count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum of the feature `x`.
- Round your answers to the seventh decimal place.

### Part (b) - Bin Width Recommendation (10 points)

- Employ the Shimazaki and Shinomoto (2007) method to suggest a bin width.
- Consider the following bin widths: `0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, and 100`.
- Recommend a bin width that results in a number of bins between 10 and 100 inclusively.
- Show your calculations for full credit.

### Part (c) - Density Estimator (10 points)

- Draw the density estimator using the bin width recommended in part (b).
- Properly label all graph elements for full credit.

## Question 2: hmeq.csv Partitioning (30 points)

### Part (a) - Dataset Baseline (10 points)

- Report the total number of observations in `hmeq.csv`.
- Provide the frequency distributions of `BAD`, including missing values.
- Calculate the means and standard deviations of `DEBTINC`, `LOAN`, `MORTDUE`, and `VALUE`.

### Part (b) - Simple Random Sampling (10 points)

- Apply simple random sampling to create Training (70%) and Testing (30%) partitions.
- State the number of observations in each partition, including those with missing values.
- Report the frequency distributions of `BAD` and the means and standard deviations of `DEBTINC`, `LOAN`, `MORTDUE`, and `VALUE` for each partition.

### Part (c) - Stratified Random Sampling (10 points)

- Perform stratified random sampling using `BAD` and `REASON` to define strata.
- Replace missing values in `BAD` with `99` and in `REASON` with `'MISSING'`.
- Report the frequency distributions of `BAD` and the means and standard deviations of `DEBTINC`, `LOAN`, `MORTDUE`, and `VALUE` for each partition.

## Question 3: FRAUD.csv Analysis (40 points)

### Part (a) - Empirical Fraud Rate (5 points)

- Calculate the percentage of fraud investigations found to be frauds.
- Round your answer to the fourth decimal place.

### Part (b) - Data Partitioning (10 points)

- Divide the complete observations into 80% Training and 20% Testing partitions.
- A complete observation has no missing values.
- Use the random seed `20230225` and `FRAUD` as the stratum variable.
- Report the number of observations in each partition.

### Part (c) - KNeighborsClassifier Training (10 points)

- Train the `KNeighborsClassifier` with a number of neighbors ranging from 2 to 7.
- Classify an observation as fraud if the proportion of `FRAUD = 1` among its neighbors is greater than or equal to the empirical fraud rate.
- Determine the misclassification rates for each number of neighbors in both partitions.

### Part (d) - Optimal Number of Neighbors (5 points)

- Identify the number of neighbors that provides the lowest misclassification rate in the Testing partition.
- In case of ties, choose the smallest number of neighbors.

### Part (e) - Prediction for a Focal Observation (10 points)

- For the focal observation with specific `DOCTOR_VISITS`, `MEMBER_DURATION`, `NUM_CLAIMS`, `NUM_MEMBERS`, `OPTOM_PRESC`, and `TOTAL_SPEND` values:
- Use the selected model from Part (d) to find its neighbors and their observation values.
- Calculate the predicted probability that this observation is a fraud.

