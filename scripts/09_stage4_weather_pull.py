"""
Stage 4 - pull Open-Meteo historical weather for each race's date/location, and
cross-reference it against FastF1's own weather channel (already pulled in
Stage 0 - complete for all 150 races, no gaps to fill, per the check that
preceded this script). Open-Meteo here is a genuine independent-source sanity
check, not a gap-filler, since FastF1's own weather data turned out not to need
one for this dataset.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from src.circuit_coordinates import CIRCUIT_COORDINATES
from src.data_loading import load_all_weather
from src.paths import DERIVED_DIR

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_open_meteo_day(lat: float, lon: float, date: str) -> dict:
    resp = requests.get(OPEN_METEO_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "hourly": "temperature_2m,precipitation,relative_humidity_2m",
        "timezone": "auto",
    }, timeout=20)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    return {
        "om_temp_mean": sum(data["temperature_2m"]) / len(data["temperature_2m"]),
        "om_temp_max": max(data["temperature_2m"]),
        "om_precip_total_mm": sum(data["precipitation"]),
        "om_humidity_mean": sum(data["relative_humidity_2m"]) / len(data["relative_humidity_2m"]),
    }


def main():
    cal = pd.read_csv(DERIVED_DIR / "all_calendars_2018_2026.csv")
    top20 = pd.read_csv(DERIVED_DIR / "top20_circuits.csv")
    cal = cal[cal["location"].isin(top20["location"])].copy()
    from src.util import slugify
    cal["circuit"] = cal["location"].apply(slugify)

    fastf1_weather = load_all_weather()
    ff1_agg = fastf1_weather.groupby(["circuit", "year", "round"]).agg(
        ff1_airtemp_mean=("AirTemp", "mean"), ff1_airtemp_max=("AirTemp", "max"),
        ff1_humidity_mean=("Humidity", "mean"),
        ff1_rainfall_any=("Rainfall", "any"),
    ).reset_index()

    rows = []
    n = len(cal)
    for i, (_, row) in enumerate(cal.iterrows(), 1):
        circuit, year, round_number, date = row["circuit"], row["year"], row["round"], row["event_date"]
        if circuit not in CIRCUIT_COORDINATES or pd.isna(date):
            continue
        date_str = str(date)[:10]
        lat, lon = CIRCUIT_COORDINATES[circuit]
        try:
            om = fetch_open_meteo_day(lat, lon, date_str)
            rows.append({"circuit": circuit, "year": year, "round": round_number, "date": date_str, **om})
        except Exception as e:
            print(f"[{i}/{n}] {circuit} {year}: FAILED - {type(e).__name__}: {e}")
        if i % 20 == 0:
            print(f"[{i}/{n}] fetched...")

    om_df = pd.DataFrame(rows)
    merged = om_df.merge(ff1_agg, on=["circuit", "year", "round"], how="left")
    merged["temp_diff_om_minus_ff1"] = merged["om_temp_mean"] - merged["ff1_airtemp_mean"]

    out_path = DERIVED_DIR / "weather_cross_reference.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n{len(merged)} races cross-referenced -> {out_path}")
    print(f"\nMean air temp difference (Open-Meteo day-average minus FastF1 race-window average): "
          f"{merged['temp_diff_om_minus_ff1'].mean():.2f}C (std {merged['temp_diff_om_minus_ff1'].std():.2f}C)")
    print("(A nonzero mean is EXPECTED, not an error: Open-Meteo's number is a whole-day average "
          "while FastF1's is measured only during the race window, which for an afternoon race is "
          "usually the warmest part of the day.)")

    # Rain agreement: does Open-Meteo's day-level precipitation total agree with
    # FastF1's own Rainfall-during-the-race flag?
    merged["om_rain_flag"] = merged["om_precip_total_mm"] > 0.5
    agree = (merged["om_rain_flag"] == merged["ff1_rainfall_any"]).mean()
    print(f"\nRain flag agreement (Open-Meteo day had >0.5mm vs FastF1 saw any rain during the race): "
          f"{agree*100:.1f}%")
    disagree = merged[merged["om_rain_flag"] != merged["ff1_rainfall_any"]]
    print(f"{len(disagree)} disagreements (expected: Open-Meteo can show rain on a day when none fell "
          f"during the specific race window, or vice versa)")


if __name__ == "__main__":
    main()
