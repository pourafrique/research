# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from CSV
data = pd.read_csv('tea_yield_data.csv')

# Split the dataset into features (X) and target (y)
X = data[['year', 'month', 'rainfall', 'temperature_max', 'temperature_min', 'hail_damage', 'SWD']]
y = data['yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM regressor
svm_regressor = SVR()

# Define a dictionary of hyperparameters to tune
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1]
}

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(estimator=svm_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from Grid Search
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Use the best hyperparameters to create the final SVM model
final_svm_regressor = SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])

# Train the final SVM model on the training data
final_svm_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = final_svm_regressor.predict(X_test)

# Evaluate the final model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# You can now use the final trained model (final_svm_regressor) to make predictions on new data.
