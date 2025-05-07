import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../alzheimers_prediction_dataset.csv')

print("Dataset Overview")
print("Number of records:", data.shape[0])
print("Number of features:", data.shape[1])
print("Columns:", data.columns.tolist())

print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nData Information:")
data.info()

print("\nMissing Values in Each Column:")
print(data.isnull().sum())

print("\nDescriptive Statistics for Numerical Features:")
print(data.describe())

duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicates}")

numerical_features = ['Age', 'Education Level', 'BMI', 'Cognitive Test Score']
for feature in numerical_features:
    if feature in data.columns:
        plt.figure(figsize=(8, 4))
        plt.hist(data[feature].dropna(), bins=30, edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

categorical_features = [col for col in data.columns if data[col].dtype == 'object']
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    data[feature].value_counts().plot(kind='bar', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
