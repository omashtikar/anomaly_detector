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
