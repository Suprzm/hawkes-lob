# Calibrated Hawkes parameters for ETH/BTC (Binance)
# Fitted on 1000 trades over 114 minutes — 0.1% intensity error
# See notebooks/03_real_data.ipynb for calibration details

CALIBRATED_PARAMS = {
    'symbol':          'ETH/BTC',
    'n_trades':        1000,
    'window_seconds':  6841.7,
    'mu':              0.0812,   # baseline intensity (trades/second)
    'alpha':           25.4618,  # jump size after each trade
    'beta':            57.1890,  # decay speed
    'eta':             0.4452,   # branching ratio = alpha/beta
    'mean_intensity':  0.1463,   # E[lambda] = mu / (1 - eta)
}