import pandas as pd

data = pd.read_csv("preprocessed_dataset.csv")
data_fe = data.copy()

binary_columns = [
    'Diabetes', 'Hypertension', 'Cholesterol Level',
    'Family History of Alzheimer’s', 'Genetic Risk Factor (APOE-ε4 allele)', "Urban vs Rural Living",
    "Alzheimer’s Diagnosis"
]
binary_mapping = {'false': 0, 'true': 1, 'no': 0, 'yes': 1, 'normal': 0, 'high': 1, 'rural': 0, 'urban': 1,}

for col in binary_columns:
    if col in data_fe.columns:
        data_fe[col] = data_fe[col].astype(str).str.lower().map(binary_mapping)
        print(f"Encoded binary feature: {col}")

ordinal_mappings = {
    'Physical Activity Level': {'low': 0, 'medium': 1, 'high': 2},
    'Alcohol Consumption': {'never': 0, 'occasionally': 1, 'regularly': 2},
    'Depression Level': {'low': 0, 'medium': 1, 'high': 2},
    'Sleep Quality': {'poor': 0, 'average': 1, 'good': 2},
    'Dietary Habits': {'unhealthy': 0, 'average': 1, 'healthy': 2},
    'Air Pollution Exposure': {'low': 0, 'medium': 1, 'high': 2},
    'Social Engagement Level': {'low': 0, 'medium': 1, 'high': 2},
    'Income Level': {'low': 0, 'medium': 1, 'high': 2},
    'Stress Levels': {'low': 0, 'medium': 1, 'high': 2}
}

for col, mapping in ordinal_mappings.items():
    if col in data_fe.columns:
        data_fe[col] = data_fe[col].astype(str).str.lower().map(mapping)
        print(f"Ordinally encoded: {col}")

nominal_columns = ['Country', 'Gender', 'Smoking Status', 'Employment Status', 'Marital Status']
data_fe = pd.get_dummies(data_fe, columns=nominal_columns, drop_first=True)
print("Applied one-hot encoding to nominal features:", nominal_columns)

if 'Age' in data_fe.columns and 'Genetic Risk Factor (APOE-ε4 allele)' in data_fe.columns:
    data_fe['Age_x_APOE'] = data_fe['Age'] * data_fe['Genetic Risk Factor (APOE-ε4 allele)']
    print("Created interaction term: Age_x_APOE")

# from sklearn.decomposition import PCA
# numerical_cols = ['Age', 'Education Level', 'BMI', 'Cognitive Test Score', 'Age_x_APOE']
# pca = PCA(n_components=0.95)  # Retain 95% variance.
# data_pca = pca.fit_transform(data_fe[numerical_cols])
# print(f"PCA applied: Reduced numerical features to {data_pca.shape[1]} components.")

print("Feature engineering complete. The dataset is now ready for modeling.")
data_fe.to_csv("fe_dataset.csv")