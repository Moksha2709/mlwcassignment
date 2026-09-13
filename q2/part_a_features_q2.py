import numpy as np
import pandas as pd

# Load noisy received 16-QAM constellation samples
df = pd.read_csv('qam16_dataset.csv')

# Feature 1: Instantaneous Amplitude r (Eq. 1)
df['r'] = np.sqrt(df['rx_I']**2 + df['rx_Q']**2)

# Feature 2: Instantaneous Phase theta (Eq. 2)
# np.arctan2 maps correctly to [-pi, pi] across all 4 quadrants
df['theta'] = np.arctan2(df['rx_Q'], df['rx_I'])

# Save augmented dataset with polar features
df.to_csv('qam16_dataset_features.csv', index=False)

# preview extracted features
print("Extracted polar features (r, theta) successfully:")
print(df[['snr_db', 'symbol_id', 'rx_I', 'rx_Q', 'r', 'theta']].head())

# summary stats for sanity check
print("\nFeature Summary Statistics:")
print(df[['r', 'theta']].describe())