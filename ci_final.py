# -*- coding: utf-8 -*-
"""Customer Purchasing Behaviors Analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, silhouette_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'Customer Purchasing Behaviors.csv'
data = pd.read_csv(file_path)

# Initial data exploration
print("First 5 Rows of the Dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Visualize numeric features
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Prepare features for clustering
X_clustering = data.drop(columns=['user_id', 'region', 'purchase_amount'])
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Clustering with hyperparameter tuning
def cluster_and_evaluate(model, params, name):
    """Cluster the data, evaluate using Silhouette Score, and return the best model."""
    best_score = -1
    best_model = None
    for param in params:
        model_instance = model(**param)
        clusters = model_instance.fit_predict(X_clustering_scaled)
        score = silhouette_score(X_clustering_scaled, clusters)
        if score > best_score:
            best_score = score
            best_model = model_instance
    print(f"{name} Best Parameters: {param}")
    print(f"{name} Best Silhouette Score: {best_score:.2f}")
    return best_model, best_score

# Hyperparameter ranges
kmeans_params = [{"n_clusters": k} for k in range(2, 6)]
agglo_params = [{"n_clusters": k} for k in range(2, 6)]
gmm_params = [{"n_components": k} for k in range(2, 6)]

# Evaluate each clustering model
best_kmeans, kmeans_score = cluster_and_evaluate(KMeans, kmeans_params, "KMeans")
best_agglo, agglo_score = cluster_and_evaluate(AgglomerativeClustering, agglo_params, "Agglomerative Clustering")
best_gmm, gmm_score = cluster_and_evaluate(GaussianMixture, gmm_params, "Gaussian Mixture")

# Compare clustering models
cluster_scores = {
    "KMeans": kmeans_score,
    "Agglomerative Clustering": agglo_score,
    "Gaussian Mixture": gmm_score
}
best_model_name = max(cluster_scores, key=cluster_scores.get)
print(f"\nBest Clustering Model: {best_model_name} with Silhouette Score = {cluster_scores[best_model_name]:.2f}")

# Plot clustering comparison
plt.figure(figsize=(10, 6))
plt.bar(cluster_scores.keys(), cluster_scores.values(), color='skyblue', alpha=0.8)
plt.title("Silhouette Score Comparison")
plt.ylabel("Silhouette Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering_scaled)
best_clusters = best_kmeans.predict(X_clustering_scaled)

plt.figure(figsize=(10, 6))
for cluster in np.unique(best_clusters):
    plt.scatter(
        X_pca[best_clusters == cluster, 0],
        X_pca[best_clusters == cluster, 1],
        label=f"Cluster {cluster}",
        alpha=0.7
    )
plt.title("Best KMeans Clustering Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# Save clusters to dataset
data['cluster'] = best_clusters
print(data[['user_id', 'cluster']].head())

# Regression Task (Predict 'purchase_amount')
X_reg = data.drop(columns=['user_id', 'purchase_amount', 'region', 'cluster'])
y_reg = data['purchase_amount']
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}
reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    reg_results[name] = {"RMSE": rmse, "R2": r2}
    print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")

# Classification Task (Predict 'region')
X_clf = data.drop(columns=['user_id', 'region', 'purchase_amount', 'cluster'])
y_clf = LabelEncoder().fit_transform(data['region'])
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}
for name, model in clf_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

# End of Code
