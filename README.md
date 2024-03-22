The task was to build classification models to predict recessions using the FRED-MD Data. 
We used a supervised learning approach, with months corresponding to recessions identified by a panel of experts (and with the benefit of hindsight) supplied as the NBER data mentioned above. 
This data classifies each month as ’Recession’ or ’Expansion’.
We ran a recursive forecasting exercise, attempting to predictthe classification of each month in to ’recession’ or ’expansion’ 6 months ahead. 
In order to do this we initially compressed the data from 127 seperate series down to 8 ’indicies’ using Principal Components Analysis (PCA). 
Functions to compute the Principal Components, and fit the classification models are pre-written for you and can be found in ’FRED MD Tools.py’. 
The PCA uses transformed FRED-MD data as described in the Mcracken & Ng paper on FRED-MD. here. 
This transformed data is precomputed for you. 
We used Logistic Regression and Support Vector Classifier from the SciKit Learnlibrary to fit the models.
