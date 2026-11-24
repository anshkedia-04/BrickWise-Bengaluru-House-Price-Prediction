# BrickWise — Bengaluru Property Valuation & Market Intelligence

**BrickWise** is a Streamlit-powered residential real estate valuation and market intelligence application covering **241 Bengaluru micro-markets**, backed by a scikit-learn Multiple Linear Regression model trained on **13,320 historical Bengaluru property listings**.

## Key Features

- **Instant Property Estimator**:
  - Searchable coverage across 241 Bengaluru localities plus one-click popular hub shortcuts.
  - Estimated market value in `₹ Lakhs` and `₹ Crores`, `₹ / sq. ft.` rate, and an indicative fair valuation band.
  - Configurable **Karnataka Stamp Duty & Registration** breakdown for total acquisition cost.
  - Interactive **Built-up Area Sensitivity Curve** (`±25%` sq. ft.) and cross-locality comparison table.
  - Non-blocking layout advisories and downloadable `.txt` valuation brief.
- **Bengaluru Market Explorer** *(powered by real historical data)*:
  - Interactive BHK and price-ceiling filters across `Bengaluru_House_Data.csv`.
  - Top micro-markets ranked by median `₹ / sq. ft.` and citywide price distribution with median markers.
  - Locality deep-dive inspector with interquartile range (`P25–P75`) and sample listings.
- **Model & Methodology Transparency**:
  - Benchmark comparison (`Linear Regression` vs `Decision Tree` vs `Lasso`) and 5-fold `ShuffleSplit` cross-validation stability.
  - Live extraction of learned additive locality price weights from the model coefficients.

## Model Performance Summary

| Metric | Value |
| --- | --- |
| Algorithm | Multiple Linear Regression (`scikit-learn`) |
| Holdout Test $R^2$ | `0.8452` |
| Mean Absolute Error (MAE) | `₹ 16.59 Lakhs` |
| Root Mean Squared Error (RMSE) | `₹ 27.99 Lakhs` |
| 5-Fold ShuffleSplit CV Mean $R^2$ | `0.8184` |
| Encoded Features | `244` (`total_sqft`, `bath`, `bhk`, + `241` localities) |

## Project Structure

```text
├── .streamlit/
│   └── config.toml                # Light architectural theme & production server defaults
├── src/
│   ├── __init__.py
│   ├── predictor.py               # Model loading, validation, sensitivity & valuation logic
│   ├── market_data.py             # Dataset sanitization & locality benchmark analytics
│   └── styles.py                  # Custom PropTech CSS styling
├── tests/
│   └── test_app.py                # Unit tests for valuation & market analytics
├── app.py                         # Streamlit multi-tab application entrypoint
├── bangalore_home_price_model.pkl # Trained LinearRegression artifact
├── columns.json                   # Feature schema & 241 locality keys
├── Bengaluru_House_Data.csv       # Historical Bengaluru listings dataset
├── Dockerfile                     # Production container image with healthcheck
└── requirements.txt               # Python runtime dependencies
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Run Tests

```bash
python -m unittest discover -s tests -v
```

## Deployment

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Connect the repository in Streamlit Community Cloud with main file path `app.py`.
3. Dependencies are automatically installed from `requirements.txt` and themed via `.streamlit/config.toml`.

### Docker (Cloud Run / Render / Railway / Fly.io)

```bash
docker build -t brickwise-bengaluru .
docker run -p 8501:8501 brickwise-bengaluru
```
