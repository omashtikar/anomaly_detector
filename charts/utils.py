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


def rate_of_change_open(opens):
    """
    Convenience wrapper to compute open-to-open rate of change.

    Parameters:
        opens (list of float): Sequence of open prices.

    Returns:
        list of float: The list of open-to-open rates of change, with length len(opens) - 1.
    """
    return rate_of_change(opens)


def detect_open_anomalies_zscore(opens):
    """
    Detect anomalies in a sequence of open prices using z-score on open-to-open returns.

    An anomaly is defined as:
        |r_t| > mean(|r|) + 3 * std(|r|)

    The returned flag list is aligned to the input length (same length as `opens`).
    The first element has no prior return so it is always 0. Positions where the
    return cannot be computed (e.g., previous open is 0) are marked 0.

    Parameters:
        opens (list of float): Sequence of open prices.

    Returns:
        list of int: A list of 0/1 flags with length len(opens) where 1 indicates an anomaly at that bar.
    """
    n = len(opens)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Compute open-to-open returns
    returns = rate_of_change(opens)  # length n-1, may include None when prev == 0
    if not returns:
        return [0] * n

    # Compute anomalies on valid returns only
    valid_returns = [r for r in returns if r is not None]
    valid_flags = detect_anomalies_zscore(valid_returns) if valid_returns else []

    # Map back to original returns positions (preserve None positions as non-anomalous)
    mapped_flags = []
    j = 0
    for r in returns:
        if r is None:
            mapped_flags.append(0)
        else:
            mapped_flags.append(1 if (j < len(valid_flags) and valid_flags[j] == 1) else 0)
            j += 1

    # Align to input length by adding a leading 0 for the first bar (no prior return)
    flags_aligned = [0] + mapped_flags

    # Safety: ensure exact alignment length
    if len(flags_aligned) < n:
        flags_aligned += [0] * (n - len(flags_aligned))
    elif len(flags_aligned) > n:
        flags_aligned = flags_aligned[:n]

    return flags_aligned
