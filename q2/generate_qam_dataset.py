"""
Dataset Generation: 16-QAM over AWGN
COM 837 - ML for Wireless Communication Systems

Generates complex baseband 16-QAM symbols (I,Q in {-3,-1,1,3}), scaled so
average symbol energy = 1, transmitted over AWGN at multiple SNR values.
200 samples per constellation point per SNR value. Seed = 67.
"""

import numpy as np
import pandas as pd

np.random.seed(67)

LEVELS          = np.array([-3, -1, 1, 3])
N_PER_SYMBOL    = 200
SNR_DB_VALUES   = [0, 5, 10, 15, 20, 25, 30]

# --- Build the 16 constellation points and normalise average energy to 1 ---
const_I, const_Q = np.meshgrid(LEVELS, LEVELS)
const_I = const_I.flatten()
const_Q = const_Q.flatten()

raw_energy = np.mean(const_I**2 + const_Q**2)   # = 10 for {-3,-1,1,3} grid
scale = 1.0 / np.sqrt(raw_energy)               # normalises avg symbol energy to 1

tx_I = const_I * scale
tx_Q = const_Q * scale
symbol_ids = np.arange(16)


def build_and_save_csv(filename='qam16_dataset.csv'):
    rows = []
    for snr_db in SNR_DB_VALUES:
        snr_lin = 10 ** (snr_db / 10)
        # Es = 1 (normalised) -> N0 = Es / SNR_lin ; noise var per dim = N0/2
        n0 = 1.0 / snr_lin
        sigma = np.sqrt(n0 / 2)

        for sid in range(16):
            noise_I = np.random.randn(N_PER_SYMBOL) * sigma
            noise_Q = np.random.randn(N_PER_SYMBOL) * sigma
            rx_I = tx_I[sid] + noise_I
            rx_Q = tx_Q[sid] + noise_Q

            for k in range(N_PER_SYMBOL):
                rows.append({
                    'snr_db': snr_db,
                    'symbol_id': sid,
                    'tx_I': tx_I[sid],
                    'tx_Q': tx_Q[sid],
                    'rx_I': rx_I[k],
                    'rx_Q': rx_Q[k],
                })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df


if __name__ == "__main__":
    df = build_and_save_csv('qam16_dataset.csv')
    print(df.shape)
    print(df.groupby('snr_db').size())
    print("\nAvg symbol energy check:",
          np.mean(df.drop_duplicates('symbol_id')['tx_I']**2 +
                   df.drop_duplicates('symbol_id')['tx_Q']**2))
