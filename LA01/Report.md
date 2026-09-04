# AML Lab Assignment 1 — Environment Setup, Preprocessing and Exploratory Data Analysis

## Student Details

- **Name:** Anirudh Awasthi
- **SAP ID:** 590027460
- **Batch:** Batch-5
- **Course:** Applied Machine Learning (CSAI2017P)

---

## Objective

The objective of this lab is to set up and verify the machine learning environment, perform initial exploratory analysis on the California Housing dataset, and study the effect of feature scaling on a KNN regression model.

The assignment covers environment verification, dataset inspection, feature-scale analysis, KNN regression and reproducibility.

---

## Datasets

### California Housing Dataset

The California Housing dataset from `scikit-learn` was used for A2 and B1.

The dataset contains **20,640 rows and 8 input features**, with `MedHouseVal` as the target variable. The target represents median house value in units of **$100,000**.

The features include:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

---

# A1 — Environment Proof

The Python environment and the required machine learning libraries were verified before performing the experiments.

### Key Code

```python
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns

print("Python Version:", sys.version)
print("NumPy Version:", np.__version__)
print("Pandas Version:", pd.__version__)
print("Scikit-learn Version:", sklearn.__version__)
print("Matplotlib Version:", matplotlib.__version__)
print("Seaborn Version:", sns.__version__)
```

## Results
| Component    | Version |
| ------------ | ------: |
| Python       |  3.14.0 |
| NumPy        |   2.5.1 |
| Pandas       |   3.0.5 |
| Scikit-learn |   1.9.0 |
| Matplotlib   |  3.11.1 |
| Seaborn      |  0.13.2 |


## Observation
Version pinning ensures that code behaves consistently across different systems. Using the same versions reduces compatibility issues and makes experiments reproducible.

## A2 — First Look at the Data
The California Housing dataset was loaded using fetch_california_housing(as_frame=True) and inspected using head(), info(), describe() and isna().sum().

### Key Code
```python
from sklearn.datasets import fetch_california_housing

d = fetch_california_housing(as_frame=True)

d.frame.head()
d.frame.info()
d.frame.describe()
d.frame.isna().sum()
```

## Dataset Summary
| Property       |         Value |
| -------------- | ------------: |
| Rows           |        20,640 |
| Input Features |             8 |
| Target         | `MedHouseVal` |
| Target Unit    |      $100,000 |
| Missing Values |          None |

All nine columns, including the target, contained 20,640 non-null values.

## Feature Scale Observation
The Population feature is on a much larger numerical scale than most of the other features. Its values range from 3 to 35,682, while features such as MedInc and HouseAge have considerably smaller ranges.

This indicates that feature scaling may be important before applying distance-based machine learning algorithms.

# B1 — Scaling Changes the Answer
The effect of feature scaling was studied using KNeighborsRegressor with n_neighbors=5.

The data was split into 80% training and 20% testing data using random_state=0. The model was first trained on the raw features and then repeated inside a pipeline using StandardScaler.


## Key Code
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = d.frame.drop(columns=["MedHouseVal"])
y = d.frame["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

mae = mean_absolute_error(y_test, pred)
print(mae)
```

## Scaled Model
```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

pipe.fit(X_train, y_train)

pred_scaled = pipe.predict(X_test)

mae_scaled = mean_absolute_error(y_test, pred_scaled)

print(mae_scaled)
```

## Results
| Model                | Test MAE |
| -------------------- | -------: |
| KNN — Raw Features   |   0.8142 |
| KNN + StandardScaler |   0.4308 |

## Observation
The MAE decreased substantially after applying StandardScaler. KNN relies on distances between data points, so features with larger numerical ranges can dominate the distance calculation when raw features are used.

Scaling places the features on a comparable scale, allowing KNN to measure distances more fairly and select more appropriate neighbours. As a result, the scaled model produced considerably better predictions.


## C1 — Reproducibility Check
The notebook was restarted and executed from top to bottom using the same random_state=0.

### Results
| Configuration      | Test MAE |
| ------------------ | -------: |
| `random_state = 0` |     0.43 |
| `random_state = 1` |     0.44 |
| Difference         |     0.01 |


## Observation

All the results were reproduced exactly after restarting the kernel and running the notebook again because the same random_state=0 was used.

After changing random_state from 0 to 1, the MAE changed from 0.43 to 0.44, a difference of only 0.01. This small variation shows that reporting the MAE to two decimal places is sufficient for this experiment.

## Conclusion
This lab verified the complete machine learning environment and provided an initial understanding of the California Housing dataset.

The exploratory analysis showed that Population has a substantially larger numerical scale than several other features. The KNN experiment demonstrated why scaling matters for distance-based models: applying StandardScaler reduced the test MAE from 0.8142 to 0.4308.

The reproducibility check also showed that fixing the random state produces consistent results, while a small change in the data split can slightly affect the reported metric.