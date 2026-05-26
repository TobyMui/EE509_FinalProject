"""AQI category definitions and helpers for PM2.5.

The PM25Vision `pm25` field is the AQI value (0-500 scale), not a raw
concentration in micrograms per cubic meter. EPA category boundaries on
the AQI scale are 50, 100, 150, 200, 300.
"""

CLASSES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

CLASS_RANGES = ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"]


def pm25_to_class(aqi_value) -> int:
    """Map a PM2.5 AQI value to one of 6 category indices (0..5)."""
    v = float(aqi_value)
    if v <= 50:
        return 0
    if v <= 100:
        return 1
    if v <= 150:
        return 2
    if v <= 200:
        return 3
    if v <= 300:
        return 4
    return 5


def describe(aqi_value) -> str:
    """Return a one-line, human-readable description of an AQI value."""
    idx = pm25_to_class(aqi_value)
    return f"AQI {max(0.0, float(aqi_value)):.1f} -> {CLASSES[idx]} (range {CLASS_RANGES[idx]})"
