## ML Challenge: Fault Detection in Device Activity
# Project Description:
This project aims to build a classification model to predict fault conditions in devices based on 47 numerical features. The task involves data preprocessing, training a machine learning model, evaluating its performance, and generating a submission file with predictions for unseen test data. The target variable 'Class' indicates whether a device is operating under normal conditions (0) or exhibiting a faulty condition (1).

# Dataset
The dataset consists of two CSV files:

TRAIN.csv: Used for training the machine learning model. It contains 47 numerical features (F01-F47) and the target variable 'Class'.
TEST.csv: Used for generating predictions. It contains 47 numerical features (F01-F47) and an 'ID' column, but no 'Class' column.
# Model and Performance
A RandomForestClassifier was chosen for this task.
The model demonstrates excellent discriminative power and high performance across both classes.

# Setup and Usage
Prerequisites
To run this notebook, you will need a Python environment with the following libraries installed:

pandas
scikit-learn
matplotlib
seaborn
You can install them using pip:

pip install pandas scikit-learn matplotlib seaborn


# Running the Notebook
1) Clone the repository (if applicable) or download the notebook (.ipynb) file.//
Run all cells sequentially. The notebook is structured to perform the following steps:
Load TRAIN.csv and TEST.csv.//
Perform initial data exploration on both datasets.//
Preprocess the data (feature scaling).//
Train a RandomForestClassifier.//
Evaluate the model's performance on a validation set.//
Create submission.csv in the required format.//

# Output
The final output will be a file named submission.csv containing two columns: 'ID' and 'Class', which represents the predicted operational status for each entry in TEST.csv.
