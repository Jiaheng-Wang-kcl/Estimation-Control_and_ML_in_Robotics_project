import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use the AdaBoost model
best_model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=30),
    n_estimators=250,
    random_state=42,
    learning_rate=1.5, 
    loss='linear'
)
# Alternatively, use the RandomForest model
# best_model = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=30,
#     random_state=42
# )
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"California Housing Prediction with AdaBoost - MSE: {mse:.4f}, R2: {r2:.4f}")

# Plot predictions vs. actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'AdaBoost on California Housing (MSE: {mse:.4f}, R2: {r2:.4f})')
plt.grid(True)
plt.show()
