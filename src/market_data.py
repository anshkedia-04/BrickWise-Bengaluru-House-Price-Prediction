from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from src.predictor import BASE_DIR, format_location_name


def parse_sqft_value(raw: object) -> float | None:
    """Parse numeric or range square-footage strings (e.g., '2100 - 2850')."""
    if pd.isna(raw):
        return None
    text = str(raw).strip()
    if "-" in text:
        parts = text.split("-")
        if len(parts) == 2:
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2.0
            except ValueError:
                return None
    try:
        val = float(text)
        return val if val > 0 else None
    except ValueError:
        return None


def parse_bhk_value(raw: object) -> int | None:
    """Extract integer BHK count from strings like '3 BHK' or '4 Bedroom'."""
    if pd.isna(raw):
        return None
    match = re.match(r"^\s*(\d+)", str(raw))
    if not match:
        return None
    val = int(match.group(1))
    return val if 1 <= val <= 15 else None


def load_clean_market_data(base_dir: Path | None = None) -> pd.DataFrame:
    """Load and sanitize Bengaluru_House_Data.csv for market analytics."""
    root = Path(base_dir) if base_dir is not None else BASE_DIR
    csv_path = root / "Bengaluru_House_Data.csv"
    df = pd.read_csv(csv_path)

    df = df.copy()
    df["location_clean"] = df["location"].fillna("Other").astype(str).str.strip()
    df["location_key"] = df["location_clean"].str.lower()
    df["location_label"] = df["location_key"].apply(format_location_name)
    df["sqft_num"] = df["total_sqft"].apply(parse_sqft_value)
    df["bhk_num"] = df["size"].apply(parse_bhk_value)
    df["bath_num"] = pd.to_numeric(df["bath"], errors="coerce")
    df["price_lakhs"] = pd.to_numeric(df["price"], errors="coerce")

    cleaned = df.dropna(subset=["sqft_num", "bhk_num", "price_lakhs"]).copy()
    cleaned = cleaned[(cleaned["sqft_num"] >= 300) & (cleaned["price_lakhs"] > 0)].copy()
    # Filter extreme sqft/bhk anomalies (< 150 sqft per room) for cleaner market benchmarks
    cleaned = cleaned[(cleaned["sqft_num"] / cleaned["bhk_num"]) >= 150].copy()
    cleaned["price_per_sqft"] = (cleaned["price_lakhs"] * 100_000.0) / cleaned["sqft_num"]
    cleaned["bhk_num"] = cleaned["bhk_num"].astype(int)
    return cleaned


def get_city_benchmarks(df: pd.DataFrame) -> dict[str, float]:
    """Compute citywide summary metrics from the cleaned market dataset."""
    return {
        "listing_count": int(len(df)),
        "median_price_lakhs": round(float(df["price_lakhs"].median()), 2),
        "median_ppsqft": round(float(df["price_per_sqft"].median()), 0),
        "median_sqft": round(float(df["sqft_num"].median()), 0),
        "unique_localities": int(df["location_key"].nunique()),
    }


def get_locality_benchmark(df: pd.DataFrame, location_key: str) -> dict[str, object] | None:
    """Return historical benchmark stats for a given lowercase locality key."""
    key = re.sub(r"\s+", " ", location_key.strip().lower())
    subset = df[df["location_key"].str.replace(r"\s+", " ", regex=True) == key]
    if subset.empty:
        return None

    return {
        "listing_count": int(len(subset)),
        "median_price_lakhs": round(float(subset["price_lakhs"].median()), 2),
        "median_ppsqft": round(float(subset["price_per_sqft"].median()), 0),
        "p25_price_lakhs": round(float(subset["price_lakhs"].quantile(0.25)), 2),
        "p75_price_lakhs": round(float(subset["price_lakhs"].quantile(0.75)), 2),
        "common_bhk": int(subset["bhk_num"].mode().iloc[0]),
        "sample_listings": subset[
            ["area_type", "size", "sqft_num", "bath_num", "price_lakhs", "price_per_sqft"]
        ]
        .rename(
            columns={
                "area_type": "Area Type",
                "size": "Configuration",
                "sqft_num": "Sq. Ft.",
                "bath_num": "Bath",
                "price_lakhs": "Price (₹ L)",
                "price_per_sqft": "₹ / Sq. Ft.",
            }
        )
        .head(12),
    }


def get_top_localities_summary(
    df: pd.DataFrame,
    min_listings: int = 25,
    top_n: int = 15,
) -> pd.DataFrame:
    """Aggregate median price/sqft and listing counts for well-represented localities."""
    grouped = (
        df.groupby(["location_key", "location_label"], as_index=False)
        .agg(
            listings=("price_lakhs", "count"),
            median_price_lakhs=("price_lakhs", "median"),
            median_ppsqft=("price_per_sqft", "median"),
            median_sqft=("sqft_num", "median"),
        )
    )
    filtered = grouped[grouped["listings"] >= min_listings].copy()
    filtered = filtered.sort_values("median_ppsqft", ascending=False).head(top_n)
    filtered["median_ppsqft"] = filtered["median_ppsqft"].round(0)
    filtered["median_price_lakhs"] = filtered["median_price_lakhs"].round(2)
    return filtered
