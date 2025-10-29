from typing import List, Dict, Any, Optional


def window_start(ts: float, interval_s: int = 10) -> int:
    """
    Floor a UNIX timestamp to the start of its interval bucket.

    Example: ts=1730000123.4, interval_s=10 -> 1730000120
    """
    return int(ts // interval_s) * interval_s


def group_ticks_by_window(ticks: List[Dict[str, Any]], interval_s: int = 10) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group already-sorted ticks into interval buckets keyed by start timestamp.

    Assumes ticks are clean and sorted ascending by 'ts'.
    """
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for t in ticks:
        start = window_start(float(t["ts"]), interval_s)
        buckets.setdefault(start, []).append(t)
    return buckets


def aggregate_window_to_ohlcv(start_ts: int, ticks: List[Dict[str, Any]], interval_s: int = 10) -> Optional[Dict[str, Any]]:
    """
    Aggregate a list of ticks that belong to the same interval into an OHLCV bar.

    Returns None if ticks is empty.
    """
    if not ticks:
        return None

    # Ticks are sorted; open is first price, close is last price
    open_price = float(ticks[0]["price"])  # type: ignore[index]
    close_price = float(ticks[-1]["price"])  # type: ignore[index]
    high_price = max(float(t["price"]) for t in ticks)
    low_price = min(float(t["price"]) for t in ticks)
    volume_sum = sum(float(t.get("volume", 0.0)) for t in ticks)

    return {
        "start": int(start_ts),
        "end": int(start_ts + interval_s),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume_sum,
    }


def ticks_to_ohlcv(ticks: List[Dict[str, Any]], interval_s: int = 10) -> List[Dict[str, Any]]:
    """
    Convert sorted ticks into a list of OHLCV bars for the given interval.
    """
    if not ticks:
        return []

    grouped = group_ticks_by_window(ticks, interval_s=interval_s)
    bars: List[Dict[str, Any]] = []
    for start in sorted(grouped.keys()):
        bar = aggregate_window_to_ohlcv(start, grouped[start], interval_s=interval_s)
        if bar is not None:
            bars.append(bar)
    return bars


__all__ = [
    "window_start",
    "group_ticks_by_window",
    "aggregate_window_to_ohlcv",
    "ticks_to_ohlcv",
]

