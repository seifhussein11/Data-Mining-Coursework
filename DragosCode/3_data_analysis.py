import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

data = pd.read_csv("preprocessed_dataset.csv")

numerical_features = ['Age', 'Education Level', 'BMI', 'Cognitive Test Score']

for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    plt.hist(data[feature].dropna(), bins=30, edgecolor='black', density=True, alpha=0.6)

    density = gaussian_kde(data[feature].dropna())
    xs = np.linspace(data[feature].min(), data[feature].max(), 200)
    plt.plot(xs, density(xs))

    plt.title(f'Distribution and Density of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()

correlation_matrix = data[numerical_features].corr()
plt.figure(figsize=(6, 5))
plt.imshow(correlation_matrix, cmap='viridis', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(numerical_features)), numerical_features, rotation=45)
plt.yticks(range(len(numerical_features)), numerical_features)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

categorical_features = [
    'Cholesterol Level', 'Diabetes',
    'Family History of Alzheimer’s', 'Hypertension',
    'Genetic Risk Factor (APOE-ε4 allele)'
]

for feature in categorical_features:
    counts = data.groupby([feature, "Alzheimer’s Diagnosis"]).size().unstack(fill_value=0)
    counts.plot(kind='bar', edgecolor='black')
    plt.title(f'{feature} vs Alzheimer\'s Diagnosis')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

plt.figure(figsize=(8, 4))
data["Alzheimer’s Diagnosis"].value_counts().plot(kind='bar', edgecolor='black')
plt.title("Distribution of Alzheimer's Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()
