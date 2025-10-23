import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Import and Preprocess the Dataset
# Load the dataset
try:
    df = pd.read_csv("Housing.csv")
except FileNotFoundError:
    print("Error: 'Housing.csv' not found. Ensure the file is in the same directory.")
    exit()

print("Initial Data Head:")
print(df.head())
print("-" * 50)


# --- Preprocessing for Multiple Linear Regression (MLR) ---

# Convert binary categorical variables ('yes'/'no') to 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# Convert nominal categorical variable 'furnishingstatus' using one-hot encoding
# We use drop_first=True to avoid multicollinearity
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)

# Define the target variable
y = df['price']


# SIMPLE LINEAR REGRESSION (SLR): Price vs. Area
print("\n" + "=" * 50)
print("SIMPLE LINEAR REGRESSION (SLR): Price vs. Area")
print("=" * 50)

# 1. Select X for SLR
X_slr = df[['area']]

# 2. Split data into train-test sets (70% train, 30% test)
X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(
    X_slr, y, test_size=0.3, random_state=42
)

# 3. Fit the Linear Regression model
slr_model = LinearRegression()
slr_model.fit(X_train_slr, y_train_slr)

# Make predictions
y_pred_slr = slr_model.predict(X_test_slr)

# 4. Evaluate model
mae_slr = mean_absolute_error(y_test_slr, y_pred_slr)
mse_slr = mean_squared_error(y_test_slr, y_pred_slr)
r2_slr = r2_score(y_test_slr, y_pred_slr)

print(f"MAE: {mae_slr:.2f}")
print(f"MSE: {mse_slr:.2f}")
print(f"R² Score: {r2_slr:.4f}")

# 5. Interpret coefficients
print(f"\nCoefficient (Area): {slr_model.coef_[0]:.2f}")
print(f"Intercept: {slr_model.intercept_:.2f}")

# 5. Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test_slr, y_test_slr, color='blue', label='Actual Prices')
plt.plot(X_test_slr, y_pred_slr, color='red', linewidth=3, label='Regression Line')
plt.title('SLR: Price vs. Area (Test Set)')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('slr_regression_line.png')
plt.show() # Use plt.show() in an IDE like PyCharm to display the plot

# MULTIPLE LINEAR REGRESSION (MLR): All Features
print("\n" + "=" * 50)
print("MULTIPLE LINEAR REGRESSION (MLR): All Features")
print("=" * 50)

# 1. Select X for MLR (all columns except 'price')
X_mlr = df.drop('price', axis=1)

# 2. Split data into train-test sets
X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
    X_mlr, y, test_size=0.3, random_state=42
)

# 3. Fit the Linear Regression model
mlr_model = LinearRegression()
mlr_model.fit(X_train_mlr, y_train_mlr)

# Make predictions
y_pred_mlr = mlr_model.predict(X_test_mlr)

# 4. Evaluate model
mae_mlr = mean_absolute_error(y_test_mlr, y_pred_mlr)
mse_mlr = mean_squared_error(y_test_mlr, y_pred_mlr)
r2_mlr = r2_score(y_test_mlr, y_pred_mlr)

print(f"MAE: {mae_mlr:.2f}")
print(f"MSE: {mse_mlr:.2f}")
print(f"R² Score: {r2_mlr:.4f}")

# 5. Interpret coefficients (Display them sorted by magnitude)
coefficients_df = pd.DataFrame({
    'Feature': X_mlr.columns,
    'Coefficient': mlr_model.coef_
})

print("\nCoefficients for Multiple Linear Regression (Sorted):")
print(coefficients_df.sort_values(by='Coefficient', ascending=False).to_markdown(index=False))

# 5. Plot Actual vs. Predicted values for MLR (for visual model assessment)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_mlr, y_pred_mlr, alpha=0.6)
# Plot the ideal line where Actual = Predicted
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit')
plt.title('MLR: Actual vs. Predicted Prices (Test Set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.savefig('mlr_actual_vs_predicted.png')
plt.show() # Use plt.show() in an IDE like PyCharm to display the plot