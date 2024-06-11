import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn import datasets

# Load dataset
lis = datasets.fetch_openml(data_id=43946)

# Perform one-hot encoding
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), [2, 3, 16])], remainder="passthrough")
new_data = ct.fit_transform(lis.data)

def knn_learning_curve(data, target, k_values):
    knn_rmse_values = []
    knn_learning_curve_data = {}

    for k in k_values:
        parameter_grid = [{"n_neighbors": [k]}]
        tuned_knn = GridSearchCV(KNeighborsRegressor(), parameter_grid, scoring="neg_root_mean_squared_error", cv=5)
        knn_scores = cross_val_score(tuned_knn, data, target, cv=10, scoring="neg_root_mean_squared_error")
        mean_rmse = -knn_scores.mean()
        knn_rmse_values.append(mean_rmse)

        train_sizes, train_scores, test_scores = learning_curve(tuned_knn, data, target, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=10, scoring="neg_root_mean_squared_error")
        knn_learning_curve_data[k] = (train_sizes, train_scores, test_scores)

        plt.plot(train_sizes, abs(test_scores.mean(axis=1)), label=f'k={k}')

    plt.xlabel('Training Examples')
    plt.ylabel('Mean Test Scores')
    plt.title('KNN Learning Curve')
    plt.legend()
    plt.show()

    return knn_rmse_values, knn_learning_curve_data

def print_rmse_table(k_values, rmse_values):
    results_df = pd.DataFrame({'k': k_values, 'RMSE': rmse_values})
    print("RMSE for Last Point of Learning Curves:")
    print(results_df)

# Usage
k_values = [5, 7, 9]
knn_rmse_values, knn_learning_curve_data = knn_learning_curve(new_data, lis.target, k_values)
print_rmse_table(k_values, knn_rmse_values)

# Define k values to test for k-nearest neighbor regressor
k_values = [1,3,5,7,9]

# Initialize lists to store RMSE for KNN and decision tree regressor
knn_rmse_values = []
dt_rmse_values = []

# Initialize linear regression model
lr = LinearRegression()

# Perform 10-fold cross-validation for linear regression
lr_scores = cross_val_score(lr, new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

# Calculate mean RMSE for linear regression
lr_mean_rmse = -lr_scores.mean()

# Iterate over k values for k-nearest neighbor regressor
for k in k_values:
    # Define parameter grid for grid search for KNN
    parameter_grid_knn = [{"n_neighbors": [k]}]

    # Perform grid search to find optimal k value for KNN
    tuned_knn = GridSearchCV(KNeighborsRegressor(), parameter_grid_knn, scoring="neg_root_mean_squared_error", cv=5)

    # Perform 10-fold cross-validation and store RMSE values for KNN
    knn_scores = cross_val_score(tuned_knn, new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

    # Calculate mean RMSE for KNN
    mean_rmse_knn = -knn_scores.mean()

    # Append mean RMSE to list for KNN
    knn_rmse_values.append(mean_rmse_knn)

    # Define parameter grid for grid search for decision tree regressor
    parameter_grid_dt = [{"min_samples_leaf": [k]}]

    # Perform grid search to find optimal min_samples_leaf value for decision tree
    tuned_dt = GridSearchCV(DecisionTreeRegressor(), parameter_grid_dt, scoring="neg_root_mean_squared_error", cv=5)

    # Perform 10-fold cross-validation and store RMSE values for decision tree
    dt_scores = cross_val_score(tuned_dt, new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

    # Calculate mean RMSE for decision tree
    mean_rmse_dt = -dt_scores.mean()

    # Append mean RMSE to list for decision tree
    dt_rmse_values.append(mean_rmse_dt)

# Plotting the learning curves
plt.figure(figsize=(10, 6))

# Plot learning curve for KNN
plt.plot(k_values, knn_rmse_values, label='KNN')

# Plot learning curve for Decision Tree
plt.plot(k_values, dt_rmse_values, label='Decision Tree')

# Plot learning curve for Linear Regression
plt.axhline(y=lr_mean_rmse, color='r', linestyle='-', label='Linear Regression')

# Plot settings
plt.xlabel('Parameter Value')
plt.ylabel('RMSE')
plt.title('Comparison of Models Learning Curves')
plt.legend()
plt.show()

# Create a table to show RMSE for the last point of learning curves
last_learning_curve_rmse = pd.DataFrame({'KNN': [knn_rmse_values[-1]],
                                         'Decision Tree': [dt_rmse_values[-1]],
                                         'Linear Regression': [lr_mean_rmse]})

print("\nRMSE for the last point of learning curves:")
print(last_learning_curve_rmse)
