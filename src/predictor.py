from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover
    InconsistentVersionWarning = UserWarning

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_METRICS = {
    "algorithm": "Multiple Linear Regression",
    "r2_score": 0.8452,
    "mae_lakhs": 16.59,
    "rmse_lakhs": 27.99,
    "cv_r2_mean": 0.8184,
    "cv_r2_scores": [0.8243, 0.7717, 0.8509, 0.8084, 0.8365],
    "raw_listings": 13320,
    "encoded_features": 244,
    "localities_count": 241,
}

MODEL_BENCHMARKS = [
    {"Model": "Linear Regression", "GridSearch CV R²": 0.8190, "Selected": "Yes"},
    {"Model": "Decision Tree Regressor", "GridSearch CV R²": 0.7218, "Selected": "No"},
    {"Model": "Lasso Regression", "GridSearch CV R²": 0.6874, "Selected": "No"},
]

CANDIDATE_POPULAR_LOCALITIES = [
    "whitefield",
    "indira nagar",
    "koramangala",
    "hsr layout",
    "electronic city",
    "hebbal",
    "jp nagar",
    "marathahalli",
    "bellandur",
    "yelahanka",
]

_UPPERCASE_TOKENS = {
    "aecs",
    "beml",
    "btm",
    "cv",
    "epip",
    "gm",
    "hal",
    "hbr",
    "hrbr",
    "hsr",
    "isro",
    "itpl",
    "jp",
    "kr",
    "lb",
    "mico",
    "ngr",
    "nri",
    "ombr",
    "rt",
    "tc",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
}


@dataclass(frozen=True)
class ValuationResult:
    """Structured output for a Bengaluru property valuation."""

    location_key: str
    location_label: str
    sqft: float
    bhk: int
    bath: int
    estimated_lakhs: float
    raw_lakhs: float
    was_clamped: bool
    estimated_crores: float
    price_per_sqft: float
    range_low_lakhs: float
    range_high_lakhs: float
    stamp_duty_lakhs: float
    registration_lakhs: float
    total_acquisition_lakhs: float
    advisories: list[str]


def format_location_name(raw_name: str) -> str:
    """Convert lowercase column keys into clean, readable locality names."""
    collapsed = re.sub(r"\s+", " ", raw_name.strip())
    words: list[str] = []
    for token in collapsed.split(" "):
        clean_alpha = token.replace(".", "").lower()
        if clean_alpha in _UPPERCASE_TOKENS:
            words.append(token.upper())
        else:
            words.append(token.capitalize())
    return " ".join(words)


def load_artifacts(base_dir: Path | None = None) -> tuple[object, list[str], list[str]]:
    """Load the serialized regression model and feature column schema."""
    root = Path(base_dir) if base_dir is not None else BASE_DIR
    model_path = root / "bangalore_home_price_model.pkl"
    columns_path = root / "columns.json"

    with columns_path.open("r", encoding="utf-8") as f:
        data_columns: list[str] = json.load(f)["data_columns"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        model = joblib.load(model_path)

    locations = data_columns[3:]
    return model, data_columns, locations


def get_popular_localities(locations: list[str]) -> list[str]:
    """Return curated popular Bengaluru micro-markets that exist in the model."""
    loc_set = set(locations)
    return [loc for loc in CANDIDATE_POPULAR_LOCALITIES if loc in loc_set]


def validate_property_inputs(sqft: float, bhk: int, bath: int) -> list[str]:
    """Return non-blocking layout advisories for atypical input combinations."""
    advisories: list[str] = []
    sqft_per_bhk = sqft / max(bhk, 1)
    if sqft_per_bhk < 300:
        advisories.append(
            f"Compact layout ({sqft_per_bhk:,.0f} sq.ft/BHK): typical Bengaluru apartments "
            "provide at least ~350–450 sq.ft per bedroom, so linear extrapolation may understate value."
        )
    elif sqft_per_bhk > 1800:
        advisories.append(
            f"Ultra-spacious layout ({sqft_per_bhk:,.0f} sq.ft/BHK): large luxury or villa-scale "
            "footprints can carry bespoke premiums not captured by standard linear pricing."
        )

    if bath > bhk + 2:
        advisories.append(
            f"Bathroom count ({bath}) significantly exceeds BHK ({bhk}); verify if utility/staff "
            "baths are included."
        )
    return advisories


def predict_price(
    model: object,
    data_columns: list[str],
    location: str,
    sqft: float,
    bhk: int,
    bath: int,
    stamp_duty_pct: float = 5.6,
    registration_pct: float = 1.0,
) -> ValuationResult:
    """Predict residential property value in Lakhs and compute acquisition breakdown."""
    x = np.zeros(len(data_columns), dtype=float)
    x[0] = float(sqft)
    x[1] = float(bath)
    x[2] = float(bhk)

    loc_key = location.strip().lower()
    if loc_key in data_columns:
        x[data_columns.index(loc_key)] = 1.0

    feature_names = list(getattr(model, "feature_names_in_", data_columns))
    features_df = pd.DataFrame([x], columns=feature_names)
    raw_lakhs = float(model.predict(features_df)[0])
    was_clamped = raw_lakhs < 5.0
    estimated_lakhs = max(raw_lakhs, 5.0)
    if was_clamped:
        advisories = validate_property_inputs(sqft, bhk, bath) + [
            "Model linear extrapolation fell below ₹5 Lakhs for this extreme input combination; "
            "displaying a minimum floor estimate."
        ]
    else:
        advisories = validate_property_inputs(sqft, bhk, bath)

    # Combine relative +/-12% spread with model MAE floor for a realistic fair band
    half_band = max(estimated_lakhs * 0.12, min(MODEL_METRICS["mae_lakhs"], estimated_lakhs * 0.35))
    range_low = max(5.0, estimated_lakhs - half_band)
    range_high = estimated_lakhs + half_band

    price_per_sqft = (estimated_lakhs * 100_000.0) / max(float(sqft), 1.0)
    stamp_duty_lakhs = estimated_lakhs * (stamp_duty_pct / 100.0)
    registration_lakhs = estimated_lakhs * (registration_pct / 100.0)
    total_acquisition_lakhs = estimated_lakhs + stamp_duty_lakhs + registration_lakhs

    return ValuationResult(
        location_key=loc_key,
        location_label=format_location_name(loc_key),
        sqft=float(sqft),
        bhk=int(bhk),
        bath=int(bath),
        estimated_lakhs=round(estimated_lakhs, 2),
        raw_lakhs=round(raw_lakhs, 2),
        was_clamped=was_clamped,
        estimated_crores=round(estimated_lakhs / 100.0, 2),
        price_per_sqft=round(price_per_sqft, 0),
        range_low_lakhs=round(range_low, 2),
        range_high_lakhs=round(range_high, 2),
        stamp_duty_lakhs=round(stamp_duty_lakhs, 2),
        registration_lakhs=round(registration_lakhs, 2),
        total_acquisition_lakhs=round(total_acquisition_lakhs, 2),
        advisories=advisories,
    )


def build_sqft_sensitivity(
    model: object,
    data_columns: list[str],
    location: str,
    sqft: float,
    bhk: int,
    bath: int,
) -> list[dict[str, float]]:
    """Generate a ±25% built-up area sensitivity curve around the user's input."""
    factors = [0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25]
    points: list[dict[str, float]] = []
    for factor in factors:
        area = max(300.0, round(sqft * factor, -1))
        res = predict_price(model, data_columns, location, area, bhk, bath)
        points.append(
            {
                "sqft": area,
                "estimated_lakhs": res.estimated_lakhs,
                "price_per_sqft": res.price_per_sqft,
                "is_selected": factor == 1.0,
            }
        )
    return points


def get_locality_coefficients(model: object, data_columns: list[str]) -> list[dict[str, object]]:
    """Extract sorted locality additive weights (in Lakhs) from the linear model."""
    coefs = getattr(model, "coef_", None)
    if coefs is None or len(coefs) != len(data_columns):
        return []

    rows = []
    for loc, coef in zip(data_columns[3:], coefs[3:]):
        rows.append(
            {
                "location_key": loc,
                "locality": format_location_name(loc),
                "premium_lakhs": round(float(coef), 2),
            }
        )
    rows.sort(key=lambda item: float(item["premium_lakhs"]), reverse=True)
    return rows
