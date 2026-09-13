import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('qam16_dataset_features.csv')
sub25 = df[df['snr_db'] == 25].reset_index(drop=True)

# ---------- 1. Elbow + Silhouette over K = 2..20 (Cartesian, SNR=25) ----------
X_cart = sub25[['rx_I', 'rx_Q']].values

K_range = range(2, 21)
inertias = []
sil_scores = []

# sweep K, record WCSS (elbow) and silhouette score for each
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_cart)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cart, labels))

# side-by-side elbow curve and silhouette curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(K_range), inertias, marker='o')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('Elbow Curve (Cartesian, SNR=25dB)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), sil_scores, marker='s', color='darkorange')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Coefficient')
axes[1].set_title('Silhouette Score (Cartesian, SNR=25dB)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part_b_elbow_silhouette.png', dpi=150)
print("Saved: part_b_elbow_silhouette.png")

elbow_df = pd.DataFrame({'K': list(K_range), 'Inertia': inertias, 'Silhouette': sil_scores})
elbow_df.to_csv('part_b_elbow_silhouette.csv', index=False)
print(elbow_df.round(3))

# ---------- 2. K=16 clustering + scatter with centroids ----------
# fit final model at K=16 (matches 16-QAM constellation size)
km16 = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
cluster_labels = km16.fit_predict(X_cart)
centroids = km16.cluster_centers_

# scatter of received points colored by cluster, with centroids marked
plt.figure(figsize=(7, 7))
scatter = plt.scatter(sub25['rx_I'], sub25['rx_Q'], c=cluster_labels,
                       cmap='tab20', s=12, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200,
            edgecolors='white', linewidths=1.5, label='Centroids')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.title('K-Means (K=16) on Cartesian Coordinates, SNR=25dB')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part_b_kmeans16_scatter.png', dpi=150)
print("Saved: part_b_kmeans16_scatter.png")


# ---------- 3. Cluster purity across 3 feature sets ----------
def cluster_purity(true_labels, cluster_assignments):
    # for each cluster, count how many points belong to its majority true symbol
    df_tmp = pd.DataFrame({'true': true_labels, 'cluster': cluster_assignments})
    correct = 0
    for c in np.unique(cluster_assignments):
        members = df_tmp[df_tmp['cluster'] == c]['true']
        if len(members) == 0:
            continue
        majority_count = members.value_counts().iloc[0]
        correct += majority_count
    return correct / len(true_labels)  # fraction of points matching cluster majority


y_true = sub25['symbol_id'].values

# compare clustering quality across Cartesian, polar, and combined features
feature_sets = {
    'Feature Set 1 (rx_I, rx_Q)':        ['rx_I', 'rx_Q'],
    'Feature Set 2 (r, theta)':          ['r', 'theta'],
    'Feature Set 3 (rx_I, rx_Q, r, theta)': ['rx_I', 'rx_Q', 'r', 'theta'],
}

purity_results = {}
for name, cols in feature_sets.items():
    X = sub25[cols].values
    if name.startswith('Feature Set 3'):
        X = StandardScaler().fit_transform(X)  # scale mixed-unit features before clustering
    km = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    purity = cluster_purity(y_true, labels)
    purity_results[name] = purity

purity_df = pd.Series(purity_results, name='Cluster Purity')
purity_df.to_csv('part_b_purity.csv')
print("\nCluster Purity by feature set (SNR=25dB, K=16):")
print(purity_df.round(4))