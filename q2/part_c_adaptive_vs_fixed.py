import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('qam16_dataset_features.csv')
SNR_VALUES = [0, 5, 10, 15, 20, 25, 30]


def cluster_purity(true_labels, cluster_assignments):
    # for each cluster, count points matching its majority true symbol
    df_tmp = pd.DataFrame({'true': true_labels, 'cluster': cluster_assignments})
    correct = 0
    for c in np.unique(cluster_assignments):
        members = df_tmp[df_tmp['cluster'] == c]['true']
        if len(members) == 0:
            continue
        correct += members.value_counts().iloc[0]
    return correct / len(true_labels)


# ---------- Strategy 1: Adaptive (fresh K-means fit per SNR) ----------
adaptive_purity = []
for snr in SNR_VALUES:
    sub = df[df['snr_db'] == snr]
    X = sub[['rx_I', 'rx_Q']].values
    y = sub['symbol_id'].values
    km = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    adaptive_purity.append(cluster_purity(y, labels))

# ---------- Strategy 2: Fixed template (fit once at SNR=25dB) ----------
sub25 = df[df['snr_db'] == 25]
X25 = sub25[['rx_I', 'rx_Q']].values
km_fixed = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
km_fixed.fit(X25)

fixed_purity = []
for snr in SNR_VALUES:
    sub = df[df['snr_db'] == snr]
    X = sub[['rx_I', 'rx_Q']].values
    y = sub['symbol_id'].values
    labels = km_fixed.predict(X)   # reuse fixed centroids, no refitting
    fixed_purity.append(cluster_purity(y, labels))

# collect and save adaptive vs fixed-template comparison
result = pd.DataFrame({
    'Adaptive K-Means': adaptive_purity,
    'Fixed Template':   fixed_purity
}, index=SNR_VALUES)
result.index.name = 'SNR (dB)'
result.to_csv('part_c_purity.csv')
print(result.round(4))

# plot purity vs SNR for both strategies
plt.figure(figsize=(7.5, 5))
plt.plot(SNR_VALUES, adaptive_purity, marker='o', label='Adaptive K-Means')
plt.plot(SNR_VALUES, fixed_purity, marker='s', label='Fixed Template (fit @ 25dB)')
plt.xlabel('SNR (dB)')
plt.ylabel('Cluster Purity')
plt.title('K=16 Demodulation: Adaptive vs Fixed-Template K-Means')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part_c_purity_vs_snr.png', dpi=150)
print("\nSaved: part_c_purity_vs_snr.png")