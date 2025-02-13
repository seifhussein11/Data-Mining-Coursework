# Data Mining Coursework Brief

**Authors:**

- Aris Karalis, ak1g24@soton.ac.uk, 36622966
- Dragos-Alexandru Gegea, dag2n24@soton.ac.uk, 36594997
- Suleyman Cagatay Demirbas, scd1r24@soton.ac.uk, 3627984
- Seif Hussein, smkh1u244@soton.ac.uk, 35639741

## Introduction

Early diagnosis of Alzheimer's disease is crucial to improve quality of life and slow the progression of symptoms. Data mining and machine learning techniques can be used to detect patterns in medical datasets that predict the presence of Alzheimer's early enough to allow interventions that may stabilize its progression [1].

## Brief

The dataset that will be used [2] contains approximately 75,000 records from 20 countries, providing insights into Alzheimer's disease risk factors. It includes various demographic, lifestyle, medical, and genetic variables and reflects the inequalities present in real-world data.

Initially, several feature engineering techniques will be applied (e.g., encoding categorical variables, standardizing numerical variables, and examining correlations). Highly correlated variables will be combined via dimensionality reduction techniques to retain most of the information rather than eliminating them outright. Additionally, new features will be created from existing ones (e.g., combining "current" smoking status with "high" physical activity) to assess their predictive importance and to get the best out of them.

Subsequently, distinct machine learning models—namely, logistic regression, random forest classifier, XGBoost classifier, and CatBoost classifier—will be applied separately to determine which ones yield the best results under cross-validation. After identifying the best-performing models, grid search will be used to optimize their parameters, and a voting classifier will be implemented to ensemble the best models to improve prediction scores. Applying class weights might be considered to mitigate bias of the dataset to have the best generalization ability.

Models will be evaluated using cross-validation with metrics such as accuracy, recall, F1-score, and ROC AUC. Once the best score is achieved, we will analyze feature importances to identify which factors most strongly predict Alzheimer's disease, presenting the results with colored graphs. Finally, feature importance will be determined using the best model's native functions, Shapley values, mutual information with the target variable, etc.

## References

1. Neelaveni, J., & Devasana, M. S. Geetha. "Alzheimer Disease Prediction using Machine Learning Algorithms." *2020 6th International Conference on Advanced Computing and Communication Systems (ICACCS)*.

2. Ankit. "Alzheimer’s Prediction Dataset (Global)." [https://www.kaggle.com/dsv/10618775](https://www.kaggle.com/dsv/10618775)
