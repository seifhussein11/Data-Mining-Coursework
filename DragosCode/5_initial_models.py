import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, StratifiedKFold

data_fe = pd.read_csv("fe_dataset.csv")

X = data_fe.drop(columns=["Alzheimer’s Diagnosis"])
y = data_fe["Alzheimer’s Diagnosis"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

validation_ratio = 0.15 / 0.85

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=validation_ratio,
    random_state=42,
    stratify=y_train_val
)

print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': LinearSVC(random_state=42)
}

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

    print(f"{name} Performance:")
    print("  Accuracy: {:.4f} ± {:.4f}".format(np.mean(cv_results['test_accuracy']),
                                               np.std(cv_results['test_accuracy'])))
    print("  F1 Score: {:.4f} ± {:.4f}".format(np.mean(cv_results['test_f1']),
                                               np.std(cv_results['test_f1'])))
    print("  ROC AUC:  {:.4f} ± {:.4f}".format(np.mean(cv_results['test_roc_auc']),
                                               np.std(cv_results['test_roc_auc'])))
    print("-" * 50)
    results[name] = cv_results

# Add these imports at the top of your file
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Create directory for saving plots if it doesn't exist
os.makedirs('confusion_matrices', exist_ok=True)

# Add this code at the end of your existing script
# Train all models on the training set and create confusion matrices
plt.figure(figsize=(20, 10))

for i, (name, model) in enumerate(models.items(), 1):
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Generate predictions on validation data
    y_val_pred = model.predict(X_val)
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    
    # Create subplot
    plt.subplot(2, 3, i)
    
    # Display the confusion matrix
    if hasattr(model, 'classes_'):
        display_labels = model.classes_
    else:
        display_labels = ['No', 'Yes']  # Default labels
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title(f"Confusion Matrix - {name}")
    
    # Save individual confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f'confusion_matrices/{name.replace(" ", "_")}_confusion_matrix.png')
    plt.close()

# Adjust layout and save combined plot
plt.figure(0)
plt.tight_layout()
plt.savefig('confusion_matrices/all_models_confusion_matrices.png')
plt.show()

print("Confusion matrices saved to 'confusion_matrices' directory")


