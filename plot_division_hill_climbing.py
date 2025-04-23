import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
file_path=r"C:\Users\AKSHITHA\OneDrive\Projects-2-2\Iai\u-1\Crop_recommendation.csv"
df=pd.read_csv(file_path)# Select relevant features 
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Encode crop labels
label_encoder = LabelEncoder()
df['crop_label'] = label_encoder.fit_transform(df['label'])
def evaluate_partition(clusters, df):
    """Compute the fitness score based on crop suitability."""
    score = 0
    for cluster in np.unique(clusters):
        cluster_data = df[clusters == cluster]
        dominant_crop = cluster_data['crop_label'].mode()[0]  # Most common crop
        score += (cluster_data['crop_label'] == dominant_crop).sum()
    return score 

def hill_climbing(X, df, k=5, max_iterations=100):
    """Optimize plot division using Hill-Climbing."""
    best_kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    best_clusters = best_kmeans.labels_
    best_score = evaluate_partition(best_clusters, df)
    
    for _ in range(max_iterations):
        new_kmeans = KMeans(n_clusters=k, n_init=10, random_state=np.random.randint(1000)).fit(X)
        new_clusters = new_kmeans.labels_
        new_score = evaluate_partition(new_clusters, df)
        
        if new_score > best_score:
            best_clusters, best_score = new_clusters, new_score
            print(f"Improved score: {best_score}")
    
    return best_clusters
optimized_clusters = hill_climbing(X_scaled, df, k=5)
def plot_clusters(X, clusters, title):
    """Plot clusters after optimization."""
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('N')
    plt.ylabel('P')
    plt.show()
plot_clusters(X_scaled, optimized_clusters, "Optimized Plot Division")