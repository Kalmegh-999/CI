import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('Customer Purchasing Behaviors.csv')

# Data preprocessing
data['high_spender'] = (data['purchase_amount'] > 300).astype(int)
X = data.drop(columns=['user_id', 'purchase_amount', 'region'])
y = data['purchase_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models for regression
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R^2': r2}

# Evaluate classification models (predicting region or high_spender)
X = data.drop(columns=['user_id', 'region'])
y = data['region']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='liblinear'),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}

for name, model in classification_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

# Clustering using KMeans and evaluating silhouette score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, clusters)
print(f"KMeans Silhouette Score: {sil_score:.2f}")

# Confusion matrix for classification evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Hyperparameter tuning for KMeans (finding optimal n_clusters)
best_silhouette = -1
best_params = {}
for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    if sil_score > best_silhouette:
        best_silhouette = sil_score
        best_params = {"n_clusters": n_clusters}

print(f"Best KMeans Params: {best_params}, Silhouette Score: {best_silhouette:.2f}")

# Logging model performance (optional)
import logging
logging.basicConfig(filename='model_performance.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Best KMeans Params: {best_params}, Silhouette Score: {best_silhouette:.2f}")
