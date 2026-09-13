import numpy as np
import pandas as pd

L = 6  # number of multipath taps

def extract_features(df):
    # extract per-tap real, imag channel coefficients and tap delays
    h_real = df[[f'h_real_{l}' for l in range(L)]].values
    h_imag = df[[f'h_imag_{l}' for l in range(L)]].values
    tau    = df[[f'tau_{l}'    for l in range(L)]].values

    amp = np.sqrt(h_real**2 + h_imag**2)          # |h_l|, shape (N, L)
    power = amp**2                                 # |h_l|^2, tap power

    # mean and std of tap amplitudes per sample
    mu    = amp.mean(axis=1, keepdims=True)
    sigma = amp.std(axis=1, keepdims=True)
    sigma_safe = np.where(sigma == 0, 1e-12, sigma)  # avoid div-by-zero

    # Kurtosis (Eq. 2): peakedness of amplitude distribution
    kurtosis = np.mean((amp - mu)**4, axis=1) / (sigma_safe.flatten()**4)

    # Skewness (Eq. 5): asymmetry of amplitude distribution
    skewness = np.mean((amp - mu)**3, axis=1) / (sigma_safe.flatten()**3)

    # Rising time (Eq. 6): delay of strongest tap minus delay of first tap (tau_0 = 0)
    strongest_idx = np.argmax(amp, axis=1)
    rising_time = tau[np.arange(len(df)), strongest_idx] - tau[:, 0]

    # RMS delay spread (Eq. 7-8): power-weighted spread of delays around mean excess delay
    total_power = power.sum(axis=1)
    total_power_safe = np.where(total_power == 0, 1e-12, total_power)
    mean_excess_delay = (tau * power).sum(axis=1) / total_power_safe
    rms_delay_spread = np.sqrt(
        ((tau - mean_excess_delay[:, None])**2 * power).sum(axis=1) / total_power_safe
    )

    # Rician K-factor (Eq. 9): ratio of dominant path power to scattered path power
    max_amp = amp.max(axis=1)
    k_factor = (max_amp**2) / (2 * sigma_safe.flatten()**2)

    # append all computed features to original dataframe
    out = df.copy()
    out['kurtosis']  = kurtosis
    out['skewness']  = skewness
    out['rising_time'] = rising_time
    out['rms_delay_spread'] = rms_delay_spread
    out['k_factor'] = k_factor
    return out


if __name__ == "__main__":
    # load raw dataset, compute features, save to new CSV
    df = pd.read_csv('los_nlos_dataset.csv')
    df_feat = extract_features(df)
    df_feat.to_csv('los_nlos_dataset_features.csv', index=False)

    # print mean feature values grouped by SNR and LOS/NLOS label
    print(df_feat[['snr_db','label','kurtosis','skewness','rising_time',
                    'rms_delay_spread','k_factor']].groupby(['snr_db','label']).mean())