# AI-ML-Internship-Task-3

** Objective**

The goal of this task is to **implement and understand Simple Linear Regression (SLR)** and **Multiple Linear Regression (MLR)** 
using **Scikit-learn**, along with proper data preprocessing, evaluation metrics, and visualization.


 **Tools & Libraries Used**

* **Python**
* **Pandas** – for data handling and preprocessing
* **NumPy** – for numerical computations
* **Matplotlib** – for data visualization
* **Scikit-learn** – for regression modeling and evaluation


 **Dataset**

The project uses the **Housing Price Prediction dataset**.
Each record contains house features such as area, number of bedrooms, availability of amenities, and the corresponding house price.


 **Steps Implemented**

#### **1. Import and Preprocess the Dataset**

* Loaded the dataset using `pandas.read_csv()`.
* Converted categorical values like `'yes'`/`'no'` into binary form (`1` and `0`).
* Applied **One-Hot Encoding** on the `'furnishingstatus'` column to convert categorical data into numerical format.
* Ensured the dataset is clean and ready for modeling.

#### **2. Simple Linear Regression (SLR)**

* Model trained on a single feature: **`area`** vs **`price`**.
* Splitted data into **train (70%)** and **test (30%)** using `train_test_split`.
* Fitted the model using `LinearRegression()`.
* Evaluated using:

  * **MAE (Mean Absolute Error)**
  * **MSE (Mean Squared Error)**
  * **R² Score**
* Visualized the **regression line** against actual data points.

#### **3. Multiple Linear Regression (MLR)**

* Model trained on **all features** to predict house price.
* Repeated the train-test split and model fitting.
* Evaluated using the same metrics (MAE, MSE, R²).
* Displayed all coefficients with their corresponding features.
* Visualized **Actual vs. Predicted prices** to assess model performance.


###  Evaluation Metrics

| Metric                        | Description                                                                         |
| :---------------------------- | :---------------------------------------------------------------------------------- |
| **MAE (Mean Absolute Error)** | Measures average magnitude of errors in predictions.                                |
| **MSE (Mean Squared Error)**  | Penalizes larger errors by squaring the differences.                                |
| **R² Score**                  | Represents how well the model explains variance in the data (closer to 1 = better). |


###  Key Learnings

* How to build and evaluate **Linear Regression models** using **Scikit-learn**.
* Importance of **data preprocessing** and **encoding categorical variables**.
* Interpretation of **regression coefficients** and **evaluation metrics**.
* Visualization of model performance and insights from regression lines.



###  Project Structure

 Linear Regression Task
│
├── Housing.csv
├── linear_regression_task3.py
├── slr_regression_line.png
├── mlr_actual_vs_predicted.png
└── README.md


