import numpy as np
import pandas as pd
from FREDMD_Tools import pca_function, fit_class_models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def calculate_brier_score(actuals, predicted):
    """
    Calculate Brier score for binary outcomes.
    """
    return np.mean((predicted - actuals) ** 2)

# Load and format data
transformed_data = pd.read_csv('2023-09-TF.csv', index_col=0, parse_dates=True).to_period('M')

# Load and format NBER data
nber_data = pd.read_csv('NBER_DATES.csv', header=None, index_col=0, parse_dates=True).to_period('M')
nber_data.columns = ['NBER']

# Convert NBER recession labels to binary format
nber_data['Recession'] = (nber_data['NBER'] == 'Recession').astype(int)

# Set up dataframe for probability results
backtest_dates = pd.date_range(start='2005-01-01', end='2021-03-01', freq='M').to_period('M')
prob_results = pd.DataFrame(index=backtest_dates, columns=['LogisticRegression', 'SVC', 'Comb'])

# Initialize variables for Brier scores
brier_scores = {'LogisticRegression': 0, 'SVC': 0, 'Comb': 0}

# Define the number of principal components
n_comps = 8

# Shift the NBER data by 6 months to match the forecasting period
shifted_nber_data = nber_data.shift(6)

# Loop through each backtest date
for date in backtest_dates:
    print(f"Processing backtest date: {date}")

    # Select the full set of time series data available at time t
    series_data = transformed_data.loc[:date]

    # Remove series with less than 36 non-missing values
    series_data = series_data.dropna(thresh=36, axis=1)

    # Standardize each series
    series_data_standardized = (series_data - series_data.mean()) / series_data.std()

    # Fill remaining missing values with zeros
    series_data_standardized.fillna(0, inplace=True)

    # Compute Principal Components
    pcs = pca_function(series_data_standardized, n_comps)

    # Convert pcs into a DataFrame with an index that matches series_data_standardized
    pcs_df = pd.DataFrame(pcs, index=series_data_standardized.index)

    # X_train holds the training data
    X_train = pcs_df

    # Align y_train with the principal components DataFrame
    y_train = shifted_nber_data.reindex(X_train.index)['Recession']

    # Fit classification models
    lr_predict, svc_predict, lr_model, svc_model = fit_class_models(X_train, y_train)

    # Store the predictions in prob_results
    prob_results.at[date, 'LogisticRegression'] = lr_predict[-1]
    prob_results.at[date, 'SVC'] = svc_predict[-1]
    prob_results.at[date, 'Comb'] = (lr_predict[-1] + svc_predict[-1]) / 2

    # Calculate Brier scores
    actual = shifted_nber_data.loc[date, 'Recession']
    if pd.isna(actual):
        print(f"Missing actual value for date {date}. Skipping.")
        continue

    for model_name in ['LogisticRegression', 'SVC', 'Comb']:
        predicted = prob_results.at[date, model_name]
        if pd.isna(predicted):
            print(f"Missing predicted value for model {model_name} on date {date}. Skipping.")
            continue
        brier_scores[model_name] += calculate_brier_score(actual, predicted)  # Pass values directly

# Normalize Brier scores
for model_name in brier_scores:
    brier_scores[model_name] /= len(backtest_dates)

# Select the best predictive model
best_model = min(brier_scores, key=brier_scores.get)

# Save the best model's probabilities to a CSV file
prob_results[[best_model]].to_csv('RecessionIndicator.csv')

print(f"Best model: {best_model} with Brier Score: {brier_scores[best_model]}")
