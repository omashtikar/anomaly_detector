import numpy as np


def rate_of_change(values):
    """
    Calculate the rate of change for a list of numbers.
    
    r_t = (i_t - i_(t-1)) / i_(t-1)
    
    Parameters:
        values (list of float): A list of numeric values.
    
    Returns:
        list of float: The list of rates of change, with length len(values) - 1.
    """
    if len(values) < 2:
        return []  # Not enough data to compute rate of change
    
    rates = []
    for t in range(1, len(values)):
        prev = values[t - 1]
        curr = values[t]
        if prev == 0:
            rates.append(None)  # Avoid division by zero
        else:
            rates.append((curr - prev) / prev)
    return rates


def detect_anomalies_zscore(rates):
    """
    Detect anomalies in a list of rates of change.

    An anomaly is defined as:
        |r_t| > mean(|r|) + 3 * std(|r|)

    Parameters:
        rates (list of float): List of rate of change values.

    Returns:
        list of int: A list of 0s and 1s where 1 indicates an anomaly.
    """
    if not rates:
        return []

    abs_rates = np.abs(rates)
    mean_abs = np.mean(abs_rates)
    std_abs = np.std(abs_rates)

    threshold = mean_abs + 3 * std_abs

    anomalies = [1 if abs(r) > threshold else 0 for r in rates]
    return anomalies




def detect_price_anomalies_zscore(prices):
    """
    Detect anomalies in tick price using z-score on tick-to-tick returns.

    Returns a 0/1 list aligned to len(prices). First element is 0.
    """
    n = len(prices)
    if n == 0:
        return []
    if n == 1:
        return [0]

    returns = rate_of_change(prices)  # may include None if prev == 0
    if not returns:
        return [0] * n

    valid_returns = [r for r in returns if r is not None]
    valid_flags = detect_anomalies_zscore(valid_returns) if valid_returns else []

    mapped_flags = []
    j = 0
    for r in returns:
        if r is None:
            mapped_flags.append(0)
        else:
            mapped_flags.append(1 if (j < len(valid_flags) and valid_flags[j] == 1) else 0)
            j += 1

    flags_aligned = [0] + mapped_flags
    if len(flags_aligned) < n:
        flags_aligned += [0] * (n - len(flags_aligned))
    elif len(flags_aligned) > n:
        flags_aligned = flags_aligned[:n]
    return flags_aligned


def detect_volume_anomalies_zscore(volumes):
    """
    Detect anomalies in tick volume using z-score on tick-to-tick volume change (rate of change).

    Returns a 0/1 list aligned to len(volumes). First element is 0.
    """
    n = len(volumes)
    if n == 0:
        return []
    if n == 1:
        return [0]

    vol_returns = rate_of_change(volumes)  # may include None if prev == 0
    if not vol_returns:
        return [0] * n

    valid = [r for r in vol_returns if r is not None]
    valid_flags = detect_anomalies_zscore(valid) if valid else []

    mapped_flags = []
    j = 0
    for r in vol_returns:
        if r is None:
            mapped_flags.append(0)
        else:
            mapped_flags.append(1 if (j < len(valid_flags) and valid_flags[j] == 1) else 0)
            j += 1

    flags_aligned = [0] + mapped_flags
    if len(flags_aligned) < n:
        flags_aligned += [0] * (n - len(flags_aligned))
    elif len(flags_aligned) > n:
        flags_aligned = flags_aligned[:n]
    return flags_aligned


def detect_price_anomalies_absmean3std(prices):
    """
    Detect anomalies in tick price using absolute-return threshold:
        |r_t| > mean(|r|) + 3 * std(|r|)

    Returns a 0/1 list aligned to len(prices). First element is 0.
    """
    n = len(prices)
    if n == 0:
        return []
    if n == 1:
        return [0]

    returns = rate_of_change(prices)
    if not returns:
        return [0] * n

    abs_returns = [abs(r) for r in returns if r is not None]
    if not abs_returns:
        return [0] * n

    mean_abs = float(np.mean(abs_returns))
    std_abs = float(np.std(abs_returns))
    thr = mean_abs + 3.0 * std_abs

    flags_returns = []
    for r in returns:
        if r is None:
            flags_returns.append(0)
        else:
            flags_returns.append(1 if abs(r) > thr else 0)

    flags = [0] + flags_returns
    if len(flags) < n:
        flags += [0] * (n - len(flags))
    elif len(flags) > n:
        flags = flags[:n]
    return flags


def detect_volume_anomalies_absmean3std(volumes):
    """
    Detect anomalies in tick volume using absolute change threshold on rate-of-change:
        |r_t| > mean(|r|) + 3 * std(|r|)

    Returns a 0/1 list aligned to len(volumes). First element is 0.
    """
    n = len(volumes)
    if n == 0:
        return []
    if n == 1:
        return [0]

    vol_returns = rate_of_change(volumes)
    if not vol_returns:
        return [0] * n

    abs_returns = [abs(r) for r in vol_returns if r is not None]
    if not abs_returns:
        return [0] * n

    mean_abs = float(np.mean(abs_returns))
    std_abs = float(np.std(abs_returns))
    thr = mean_abs + 3.0 * std_abs

    flags_returns = []
    for r in vol_returns:
        if r is None:
            flags_returns.append(0)
        else:
            flags_returns.append(1 if abs(r) > thr else 0)

    flags = [0] + flags_returns
    if len(flags) < n:
        flags += [0] * (n - len(flags))
    elif len(flags) > n:
        flags = flags[:n]
    return flags
