"""
Approximate lat/lon for each of the 20 circuits, used only to query Open-Meteo's
historical weather API for cross-referencing FastF1's own weather channel (Stage 4).
Hand-specified from public, stable, well-documented circuit locations - precise
enough for hourly-resolution regional weather (which is what Open-Meteo provides),
not claimed to be exact enough for on-track telemetry purposes.
"""

CIRCUIT_COORDINATES = {
    "austin": (30.1328, -97.6411),
    "baku": (40.3725, 49.8533),
    "barcelona": (41.5700, 2.2611),
    "budapest": (47.5789, 19.2486),
    "imola": (44.3439, 11.7167),
    "jeddah": (21.6319, 39.1044),
    "lusail": (25.4900, 51.4542),
    "melbourne": (-37.8497, 144.9680),
    "mexico_city": (19.4042, -99.0907),
    "monaco": (43.7347, 7.4206),
    "montr_al": (45.5000, -73.5228),
    "monza": (45.6156, 9.2811),
    "s_o_paulo": (-23.7036, -46.6997),
    "sakhir": (26.0325, 50.5106),
    "silverstone": (52.0786, -1.0169),
    "spa_francorchamps": (50.4372, 5.9714),
    "spielberg": (47.2197, 14.7647),
    "suzuka": (34.8431, 136.5410),
    "yas_island": (24.4672, 54.6031),
    "zandvoort": (52.3888, 4.5409),
}
