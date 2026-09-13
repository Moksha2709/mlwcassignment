import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load extracted features dataset
df = pd.read_csv('los_nlos_dataset_features.csv')

SNR_VALUES = [0, 5, 10, 15, 20, 25, 30]

# feature sets to test: 5 single-feature SVMs + 1 combined SVM
FEATURE_SETS = {
    'SVM-1 (Kurtosis)':         ['kurtosis'],
    'SVM-2 (Skewness)':         ['skewness'],
    'SVM-3 (Rising time)':      ['rising_time'],
    'SVM-4 (RMS delay spread)': ['rms_delay_spread'],
    'SVM-5 (Rician K-factor)':  ['k_factor'],
    'SVM-6 (All 5 combined)':   ['kurtosis', 'skewness', 'rising_time',
                                 'rms_delay_spread', 'k_factor'],
}

results = {name: [] for name in FEATURE_SETS}

# Train and evaluate across all SNR levels
for snr in SNR_VALUES:
    sub = df[df['snr_db'] == snr].reset_index(drop=True)
    y = sub['label'].values

    for name, cols in FEATURE_SETS.items():
        X = sub[cols].values
        
        # 80/20 train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features using training statistics only (avoids data leakage)
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Train RBF SVM with C=1
        clf = SVC(kernel='rbf', C=1)
        clf.fit(X_train_s, y_train)
        
        # Evaluate accuracy on held-out test split
        acc = accuracy_score(y_test, clf.predict(X_test_s)) * 100
        results[name].append(acc)

# Save results table: rows = SNR, columns = SVM feature set, values = accuracy %
results_df = pd.DataFrame(results, index=SNR_VALUES)
results_df.index.name = 'SNR (dB)'
results_df.to_csv('part_b_results.csv')
print("Part (b) Classification Accuracies (%):")
print(results_df.round(2))

# Plot Accuracy vs SNR for all 6 SVM classifiers
plt.figure(figsize=(8, 5.5))
markers = ['o', 's', '^', 'D', 'v', '*']
for (name, accs), m in zip(results.items(), markers):
    plt.plot(SNR_VALUES, accs, marker=m, linewidth=1.5, label=name)
plt.xlabel('SNR (dB)', fontsize=11)
plt.ylabel('Classification Accuracy (%)', fontsize=11)
plt.title('SVM Classification Accuracy vs SNR (Single & Combined Features)', fontsize=12)
plt.legend(fontsize=8.5, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part_b_accuracy_vs_snr.png', dpi=150)
print("\nSaved plot: part_b_accuracy_vs_snr.png")