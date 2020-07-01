# Mesothelioma-Diagnosis
Computational Model for Prediction of Malignant Mesothelioma Diagnosis 

Installation: To run the scripts, you need to have installed:

Spyder(Python) 
Python 3.7
Python packages panda
pip install panda
Python packages panda
pip install numpy
pip install keras
pip install tensorflow

You need to have root privileges, an internet connection, and at least 1 GB of free space on your hard disk.
Our scripts were originally developed on a Dell -15JPO9P computer with an Intel Core i7-8550U CPU 1.80GHz processor, with 8 GB of Random-Access Memory (RAM).

Dataset preparation Download the Cervical cancer (Risk Factors) Data Set file at the following URL: https://archive.ics.uci.edu/ml/datasets/Mesothelioma%C3%A2%E2%82%AC%E2%84%A2s+disease+data+set+
and update the path in the code as per location of file.

To run the Python code on unbalanced data with full set of features: Execute Mesothelioma_Imbalance.py
Python Code for all the classification algorithms: Execute method.py

Balancing Strategy Resampling
To run the python code on data balanced using resampling strategy:Execute Resample.py
For different feature sets:
For feature set selected using Genetic algorithm: Execute Resample_genetic.py
For feature set selected using OLS method: Execute Resample_ols.py
For feature set selected using Random Forest Feature Selection Technique: Execute Resample_RFFS.py
For feature set selected using PCA algorithm: Execute Resample_pca.py

Balancing Strategy SMOTE
To run the python code on data balanced using SMOTE: Execute smote.py
For different feature sets: 
For feature set selected using Genetic algorithm: Execute smote_genetic.py
For feature set selected using OLS method: Execute smote_ols.py
For feature set selected using Random Forest Feature Selection Technique: Execute smote_RFFS.py
For feature set selected using PCA algorithm: Execute smote_pca.py

Balancing Strategy ADASYN: 
To run the python code on data balanced using ADASYN: Execute adasyn.py
For different feature sets: 
For feature set selected using Genetic algorithm: Execute adasyn_genetic.py
For feature set selected using OLS method: Execute adasyn_ols.py
For feature set selected using Random Forest Feature Selection Technique: Execute adasyn_RFFS.py
For feature set selected using PCA algorithm: Execute adasyn_pca.py

The Mesotheliomaâ€™s disease data set is publically available on the website of the University of California Irvine Machine Learning Repository, under its copyright license.

Reference
More information about this project can be found on this paper:
Surbhi Gupta and Manoj K. Gupta "A Computational Model for Prediction of Malignant Mesothelioma Diagnosis".

Contacts
This sofware was developed by Surbhi Gupta at the School of Computer Science & Engineering, Shri Mata Vaishno Devi University, Sub-Post Office,  Network Centre, Katra, Jammu and Kashmir 182320, India . 
For questions or help, please write to sur7312@gmail.com 
