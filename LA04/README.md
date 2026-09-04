# Applied Machine Learning (CSAI2017P)
## Lab Assignment 4 — Customer Churn Prediction

**Name:** Anirudh Awasthi  
**SAP ID:** 590027460  
**Batch:** Batch-5  

---

## Objective

The objective of this experiment is to build a classification pipeline for customer churn prediction using mixed numerical and categorical features. The experiment focuses on preprocessing, logistic regression, confusion-matrix based evaluation, model comparison, and identifying features carrying the strongest predictive signal.

## Dataset

The **Telco Customer Churn** dataset contains 7,043 customer records with numerical and categorical features. The target variable is `Churn`, which indicates whether a customer has left the service.

The dataset was loaded using the following code:

```python
import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

```
## A1 — Encode and Fit

Categorical features were one-hot encoded and numerical features were imputed and scaled using a ColumnTransformer. Logistic Regression was then implemented inside a Pipeline.

### Result
Test Accuracy: 80.34%

Overall Churn Rate: 26.54%

### Observation
The logistic regression model achieved 80.34% test accuracy, while the overall churn rate was 26.54%. The accuracy is much higher because the dataset contains more non-churning customers, so accuracy alone does not fully represent churn detection performance.
```

```
## A2 — Confusion Matrix

The confusion matrix and classification report were used to evaluate the model's ability to identify customers who churn.

## Result
Confusion Matrix:
```
[[923 112]

 [165 209]]

Class Precision	Recall	F1-score
No Churn	0.85	0.89	0.87
Churn	    0.65	0.56	0.60

Churners missed by the model: 165
```
## Observation

The model missed 165 actual churners, which are false negatives. Missing a churner can result in lost future revenue, while offering retention incentives to some customers who would have stayed generally represents a smaller business cost.


## B1 — Model Comparison

Three classification models were evaluated using the same preprocessing pipeline.

```
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.8034	0.6511	0.5588	0.6014	0.8515
Decision Tree	    0.7935	0.6095	0.6176	0.6135	0.8368
Random Forest	    0.7864	0.6327	0.4652	0.5362  0.8256
```

### Observation

Logistic Regression achieved the highest accuracy (80.34%) and ROC-AUC (0.8515), while Decision Tree had slightly higher recall (61.76%) and F1-score (0.6135). Random Forest performed lowest on recall and F1-score. Logistic Regression was selected as the best overall model due to its stronger ROC-AUC and accuracy.


## C1 — Which Features Carry Signal?

The top 10 features were identified using the absolute values of Logistic Regression coefficients.

```
Feature	Coefficient
tenure	- 1.3895
Contract — Two year	-0.7678
TotalCharges 0.6948
Contract — Month-to-month	0.6031
InternetService — DSL	-0.3649
PhoneService — Yes	-0.3350
OnlineSecurity — Yes	-0.3240
PaperlessBilling — No	-0.3186
TechSupport — Yes	-0.2772
MultipleLines — No	-0.2710
```

### Observation

The strongest signals were tenure (-1.3895), two-year contract (-0.7678), and TotalCharges (0.6948). Month-to-month contracts also showed a strong positive coefficient (0.6031), indicating higher churn tendency. These features are generally available at the time of prediction.

## Conclusion

The experiment demonstrated a complete customer churn classification workflow using preprocessing pipelines and multiple classification models. Logistic Regression provided the strongest overall performance with 80.34% accuracy and 0.8515 ROC-AUC. The feature analysis showed that tenure, contract type, and TotalCharges carried substantial predictive signal for customer churn.