import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../alzheimers_prediction_dataset.csv')

duplicates = data.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")
if duplicates > 0:
    data = data.drop_duplicates()
    print("Duplicates dropped.")

categorical_columns = [
    'Country', 'Gender', 'Physical Activity Level', 'Smoking Status',
    'Alcohol Consumption', 'Diabetes', 'Hypertension', 'Cholesterol Level',
    'Family History of Alzheimer’s', 'Depression Level', 'Sleep Quality',
    'Dietary Habits', 'Air Pollution Exposure', 'Employment Status',
    'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels',
    'Urban vs Rural Living', "Alzheimer's Diagnosis"
]

numerical_features = ['Age', 'Education Level', 'BMI', 'Cognitive Test Score']
for feature in numerical_features:
    if feature in data.columns:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        print(f"{feature}: Found {outliers.shape[0]} outliers (values outside [{lower_bound:.2f}, {upper_bound:.2f}]).")

scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[numerical_features] = scaler.fit_transform(data[numerical_features])
print("Numerical features have been standardized.")

print("Summary statistics for scaled numerical features:")
print(data_scaled[numerical_features].describe())
data_scaled.to_csv("preprocessed_dataset.csv")
