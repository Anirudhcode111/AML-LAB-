# AML Lab Assignment 2 — Regression: Housing Price Prediction

## Student Details

- **Name:** Anirudh Awasthi
- **SAP ID:** 590027460
- **Batch:** Batch-5
- **Course:** Applied Machine Learning (CSAI2017P)


## Objective

The objective of this lab is to build and evaluate regression models for housing price prediction. The assignment covers simple and multiple linear regression, polynomial regression and overfitting, along with analysing where the model makes its largest errors.


## Dataset

**California Housing Dataset**

The California Housing dataset from scikit-learn was used for all experiments. The target variable is `MedHouseVal`, which represents the median house value in units of $100,000.


## A1 — Simple Linear Regression

The model was trained to predict MedHouseVal using only MedInc as the input feature. The data was split into 80% training and 20% testing data using random_state=0.



### Key Code
```python
X = df[["MedInc"]]

y = df["MedHouseVal"]


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
```


## Results
Slope: 0.42

Intercept: 0.44

Test R²: 0.44


## Observation
The model shows a positive relationship between MedInc and MedHouseVal, with a slope of 0.42, meaning a 1-unit increase in income is associated with about a $42,000 increase in predicted house value. The test R² of 0.44 shows that income alone explains around 44% of the variation, so other factors also matter.

-----------------------
-----------------------

## A2 — Multiple Linear Regression
The model was trained using all eight features of the California Housing dataset. A scaling pipeline with StandardScaler and LinearRegression was used, with 80% of the data used for training and 20% for testing.

## Key Code
```python
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

## Results
Test MAE: 0.53

Test RMSE: 0.72

Test R²: 0.59

## Observation

The predictions generally increase with the actual house values, showing a clear positive relationship. However, the points are fairly scattered, especially for higher-valued houses, indicating noticeable prediction errors.

---
---

## B1 — Polynomial Features and Overfitting
Polynomial regression models of degree 1, 2 and 3 were built using PolynomialFeatures followed by LinearRegression. Train and test RMSE were compared to identify where overfitting begins.

## Results
| Degree | Train RMSE | Test RMSE |
|--------|-----------:|----------:|
| 1      | 0.723492   | 0.727313  |
| 2      | 0.647843   | 1.662482  |
| 3      | 0.600468   | 228.143584 |

## Observation

Overfitting starts at degree 2, where train RMSE decreases from 0.723 to 0.648 but test RMSE increases from 0.727 to 1.662. At degree 3, the gap becomes extreme, showing that the model is fitting the training data too closely and failing to generalize.

---
---


## C1 — Where the Model Fails
The 20 test districts with the largest absolute prediction errors were identified using the best-performing model.

## Observation
#### Many of the largest errors occur for districts whose actual MedHouseVal is at the upper limit of 5.0, while the model predicts much lower values. This suggests that the target cap makes these high-value districts difficult to model accurately. More detailed property-level and location-based features could help.


---

## Conclusion

The experiments showed that using all available features improved the regression performance compared with using MedInc alone. Polynomial regression demonstrated how increasing model complexity can lead to severe overfitting, while the error analysis showed that the model struggles particularly with high-value districts.