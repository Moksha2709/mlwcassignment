import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('los_nlos_dataset_features.csv')
SNR_VALUES = [0, 5, 10, 15, 20, 25, 30]
FEATURES = ['kurtosis', 'skewness', 'rising_time', 'rms_delay_spread', 'k_factor']

# Step 1: Train once on SNR = 25 dB (80% train split)
train_snr = 25
sub_train = df[df['snr_db'] == train_snr].reset_index(drop=True)
X_all = sub_train[FEATURES].values
y_all = sub_train['label'].values

X_train, _, y_train, _ = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Fit scaler on 25 dB training data
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

# Fit SVM classifier with RBF kernel and C=1
clf_fixed = SVC(kernel='rbf', C=1)
clf_fixed.fit(X_train_s, y_train)

# Step 2: Evaluate this fixed model on the test split of every SNR
fixed_train_accs = []
matched_accs = pd.read_csv('part_b_results.csv', index_col=0)['SVM-6 (All 5 combined)'].values

for snr in SNR_VALUES:
    sub = df[df['snr_db'] == snr].reset_index(drop=True)
    X = sub[FEATURES].values
    y = sub['label'].values
    
    # Extract test split for this SNR (same split as part (b), same random_state)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Transform test set using the fixed 25 dB scaler (no refitting)
    X_test_s = scaler.transform(X_test)
    
    # Predict with the 25 dB-trained model and score against this SNR's labels
    acc = accuracy_score(y_test, clf_fixed.predict(X_test_s)) * 100
    fixed_train_accs.append(acc)

# Save comparison results: matched-SNR training vs fixed 25 dB training
result = pd.DataFrame({
    'Train = Test SNR': matched_accs,
    'Train at 25 dB':   fixed_train_accs
}, index=SNR_VALUES)
result.index.name = 'SNR (dB)'
result.to_csv('part_c_results.csv')
print("Part (c) Generalization Comparison (%):")
print(result.round(2))

# Plot Matched-SNR vs Fixed 25 dB Training
plt.figure(figsize=(7.5, 5))
plt.plot(SNR_VALUES, matched_accs, marker='o', linewidth=1.5, label='Train = Test SNR (Matched)')
plt.plot(SNR_VALUES, fixed_train_accs, marker='s', linewidth=1.5, label='Train at 25 dB (Fixed)')
plt.xlabel('SNR (dB)', fontsize=11)
plt.ylabel('Classification Accuracy (%)', fontsize=11)
plt.title('SVM-6 (5 features): Matched-SNR vs Fixed 25 dB Training', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part_c_accuracy_vs_snr.png', dpi=150)
print("\nSaved plot: part_c_accuracy_vs_snr.png")