import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

data_fe = pd.read_csv("fe_dataset.csv")

X = data_fe.drop(columns=["Alzheimer’s Diagnosis"])
y = data_fe["Alzheimer’s Diagnosis"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y  # maintains the distribution of the target
)

validation_ratio = 0.15 / 0.85

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=validation_ratio,
    random_state=42,
    stratify=y_train_val
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(
    estimator=gb,
    param_grid=param_grid_gb,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search_gb.fit(X_train, y_train)

print("Best Gradient Boosting Parameters:", grid_search_gb.best_params_)
print("Best cross-validation ROC AUC for Gradient Boosting:", grid_search_gb.best_score_)

best_gb = grid_search_gb.best_estimator_
y_val_pred_gb = best_gb.predict(X_val)
y_val_proba_gb = best_gb.predict_proba(X_val)[:, 1]

print("\nGradient Boosting Validation Performance:")
print("Accuracy:", accuracy_score(y_val, y_val_pred_gb))
print("F1 Score:", f1_score(y_val, y_val_pred_gb))
print("ROC AUC:", roc_auc_score(y_val, y_val_proba_gb))


param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search_rf.fit(X_train, y_train)

print("\nBest Random Forest Parameters:", grid_search_rf.best_params_)
print("Best cross-validation ROC AUC for Random Forest:", grid_search_rf.best_score_)

best_rf = grid_search_rf.best_estimator_
y_val_pred_rf = best_rf.predict(X_val)
y_val_proba_rf = best_rf.predict_proba(X_val)[:, 1]

print("\nRandom Forest Validation Performance:")
print("Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print("F1 Score:", f1_score(y_val, y_val_pred_rf))
print("ROC AUC:", roc_auc_score(y_val, y_val_proba_rf))
