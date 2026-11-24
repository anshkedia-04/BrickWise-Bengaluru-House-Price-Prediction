from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.market_data import (
    get_city_benchmarks,
    get_locality_benchmark,
    get_top_localities_summary,
    load_clean_market_data,
)
from src.predictor import (
    MODEL_BENCHMARKS,
    MODEL_METRICS,
    build_sqft_sensitivity,
    format_location_name,
    get_locality_coefficients,
    get_popular_localities,
    load_artifacts,
    predict_price,
)
from src.styles import CUSTOM_CSS

st.set_page_config(
    page_title="BrickWise · Bengaluru Property Valuation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading valuation model...")
def get_model_resources():
    return load_artifacts()


@st.cache_data(show_spinner="Indexing Bengaluru market listings...")
def get_market_data() -> pd.DataFrame:
    return load_clean_market_data()


def _format_valuation_report(result, locality_bench: dict | None, city_bench: dict) -> str:
    lines = [
        "BRICKWISE — BENGALURU PROPERTY VALUATION BRIEF",
        "=" * 50,
        f"Locality:              {result.location_label}",
        f"Configuration:         {result.bhk} BHK · {result.bath} Bath",
        f"Built-up Area:         {result.sqft:,.0f} sq. ft.",
        "",
        f"Estimated Value:       ₹ {result.estimated_lakhs:,.2f} Lakhs (₹ {result.estimated_crores:,.2f} Cr)",
        f"Indicative Fair Range: ₹ {result.range_low_lakhs:,.2f} – ₹ {result.range_high_lakhs:,.2f} Lakhs",
        f"Estimated Rate:        ₹ {result.price_per_sqft:,.0f} / sq. ft.",
        "",
        "Acquisition Breakdown (Indicative):",
        f"  Base Property Value: ₹ {result.estimated_lakhs:,.2f} Lakhs",
        f"  Stamp Duty:          ₹ {result.stamp_duty_lakhs:,.2f} Lakhs",
        f"  Registration Fee:    ₹ {result.registration_lakhs:,.2f} Lakhs",
        f"  Total Estimated:     ₹ {result.total_acquisition_lakhs:,.2f} Lakhs",
        "",
        "Market Benchmarks:",
        f"  Bengaluru Median:    ₹ {city_bench['median_ppsqft']:,.0f} / sq. ft. ({city_bench['listing_count']:,} listings)",
    ]
    if locality_bench:
        lines.append(
            f"  Locality Median:     ₹ {locality_bench['median_ppsqft']:,.0f} / sq. ft. "
            f"({locality_bench['listing_count']} listings)"
        )
    lines.extend(
        [
            "",
            f"Model: {MODEL_METRICS['algorithm']} | Test R² = {MODEL_METRICS['r2_score']:.4f} | "
            f"MAE = ₹ {MODEL_METRICS['mae_lakhs']:.2f} L",
            "Note: Estimates are data-driven approximations based on historical Bengaluru listings.",
        ]
    )
    return "\n".join(lines)


def _render_header(city_bench: dict, locations_count: int) -> None:
    st.markdown(
        f"""
        <div class="bw-header">
            <div class="bw-brand-row">
                <div class="bw-logo-mark">BW</div>
                <div>
                    <h1 class="bw-title">BrickWise Bengaluru</h1>
                    <p class="bw-subtitle">
                        Residential Property Valuation &amp; Micro-Market Intelligence
                    </p>
                </div>
            </div>
            <div class="bw-header-pills">
                <div class="bw-pill"><strong>{locations_count}</strong> Micro-Markets</div>
                <div class="bw-pill"><strong>{city_bench['listing_count']:,}</strong> Listings Indexed</div>
                <div class="bw-pill">Test R² <strong>{MODEL_METRICS['r2_score'] * 100:.1f}%</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_estimator_tab(
    model: object,
    data_columns: list[str],
    locations: list[str],
    market_df: pd.DataFrame,
    city_bench: dict,
) -> None:
    label_to_key = {format_location_name(loc): loc for loc in locations}
    sorted_labels = sorted(label_to_key.keys())

    default_label = "Whitefield" if "Whitefield" in label_to_key else sorted_labels[0]
    if "selected_location_label" not in st.session_state:
        st.session_state["selected_location_label"] = default_label

    popular_keys = get_popular_localities(locations)[:6]
    popular_labels = [format_location_name(k) for k in popular_keys]

    col_input, col_output = st.columns([1.05, 1.55], gap="large")

    with col_input:
        st.markdown(
            """
            <div class="bw-card">
                <p class="bw-card-title">Property Specifications</p>
                <p class="bw-card-caption">
                    Configure locality, layout, and built-up area for an instant valuation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if popular_labels:
            st.caption("Popular hubs")
            chip_cols = st.columns(3)
            for idx, hub_label in enumerate(popular_labels):
                if chip_cols[idx % 3].button(
                    hub_label,
                    key=f"hub_btn_{idx}",
                    width="stretch",
                ):
                    st.session_state["selected_location_label"] = hub_label

        selected_label = st.selectbox(
            "Bengaluru Micro-Market / Locality",
            options=sorted_labels,
            key="selected_location_label",
            help="241 Bengaluru residential micro-markets learned by the valuation model.",
        )
        selected_loc_key = label_to_key[selected_label]

        sqft = st.number_input(
            "Total Built-up Area (sq. ft.)",
            min_value=300,
            max_value=12000,
            value=1350,
            step=50,
            help="Enter total super built-up / built-up area in square feet.",
        )

        c_bhk, c_bath = st.columns(2)
        with c_bhk:
            bhk = st.selectbox("Bedrooms (BHK)", options=[1, 2, 3, 4, 5, 6], index=1)
        with c_bath:
            default_bath_idx = min(bhk - 1, 5)
            bath = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5, 6], index=default_bath_idx)

        with st.expander("Acquisition Cost Assumptions (Karnataka)", expanded=False):
            stamp_duty_pct = st.slider("Stamp Duty (%)", 2.0, 8.0, 5.6, 0.1)
            registration_pct = st.slider("Registration Fee (%)", 0.5, 2.5, 1.0, 0.1)

        area_per_bhk = sqft / max(bhk, 1)
        st.caption(f"Space efficiency: **{area_per_bhk:,.0f} sq. ft. per BHK**")

    result = predict_price(
        model=model,
        data_columns=data_columns,
        location=selected_loc_key,
        sqft=sqft,
        bhk=bhk,
        bath=bath,
        stamp_duty_pct=stamp_duty_pct,
        registration_pct=registration_pct,
    )
    locality_bench = get_locality_benchmark(market_df, selected_loc_key)

    with col_output:
        if locality_bench:
            diff_pct = (
                (result.price_per_sqft - locality_bench["median_ppsqft"])
                / max(locality_bench["median_ppsqft"], 1.0)
            ) * 100.0
            if abs(diff_pct) < 3.0:
                badge_html = '<span class="bw-badge-neutral">In line with locality median</span>'
            elif diff_pct > 0:
                badge_html = f'<span class="bw-badge-up">+{diff_pct:.1f}% vs locality median</span>'
            else:
                badge_html = f'<span class="bw-badge-down">{diff_pct:.1f}% vs locality median</span>'
        else:
            badge_html = '<span class="bw-badge-neutral">Model estimate</span>'

        st.markdown(
            f"""
            <div class="bw-valuation-hero">
                <div class="bw-eyebrow">Estimated Market Value · {result.location_label}</div>
                <p class="bw-price-main">₹ {result.estimated_lakhs:,.2f} Lakhs</p>
                <div class="bw-price-sub">
                    <span>≈ <span class="bw-mono">₹ {result.estimated_crores:,.2f} Cr</span></span>
                    <span>•</span>
                    <span>Rate <span class="bw-mono">₹ {result.price_per_sqft:,.0f} / sq. ft.</span></span>
                    <span>•</span>
                    <span>Fair band <span class="bw-mono">₹ {result.range_low_lakhs:,.1f}L – ₹ {result.range_high_lakhs:,.1f}L</span></span>
                    {badge_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for note in result.advisories:
            st.markdown(f'<div class="bw-advisory">{note}</div>', unsafe_allow_html=True)

        loc_median_str = (
            f"₹ {locality_bench['median_ppsqft']:,.0f}"
            if locality_bench
            else "N/A"
        )
        loc_note_str = (
            f"{locality_bench['listing_count']} historical listings"
            if locality_bench
            else "Limited direct history"
        )

        st.markdown(
            f"""
            <div class="bw-kpi-grid">
                <div class="bw-kpi-box">
                    <div class="bw-kpi-label">All-In Acquisition</div>
                    <div class="bw-kpi-value">₹ {result.total_acquisition_lakhs:,.2f} L</div>
                    <div class="bw-kpi-note">Incl. ₹ {result.stamp_duty_lakhs + result.registration_lakhs:,.2f}L duty &amp; reg.</div>
                </div>
                <div class="bw-kpi-box">
                    <div class="bw-kpi-label">Locality Median Rate</div>
                    <div class="bw-kpi-value">{loc_median_str} / sq.ft</div>
                    <div class="bw-kpi-note">{loc_note_str}</div>
                </div>
                <div class="bw-kpi-box">
                    <div class="bw-kpi-label">Bengaluru Median</div>
                    <div class="bw-kpi-value">₹ {city_bench['median_ppsqft']:,.0f} / sq.ft</div>
                    <div class="bw-kpi-note">Citywide median ₹ {city_bench['median_price_lakhs']:,.1f}L</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sensitivity_df = pd.DataFrame(
            build_sqft_sensitivity(
                model=model,
                data_columns=data_columns,
                location=selected_loc_key,
                sqft=sqft,
                bhk=bhk,
                bath=bath,
            )
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sensitivity_df["sqft"],
                y=sensitivity_df["estimated_lakhs"],
                mode="lines+markers",
                name="Estimated Price",
                line=dict(color="#0F172A", width=2.5),
                marker=dict(size=7, color="#0F172A"),
                hovertemplate="Area: %{x:,.0f} sq. ft.<br>Estimate: ₹ %{y:,.2f} L<extra></extra>",
            )
        )
        selected_row = sensitivity_df[sensitivity_df["is_selected"]]
        if not selected_row.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected_row["sqft"],
                    y=selected_row["estimated_lakhs"],
                    mode="markers",
                    name="Your Configuration",
                    marker=dict(size=13, color="#B84A27", line=dict(color="#FFFFFF", width=2)),
                    hovertemplate="Selected: %{x:,.0f} sq. ft.<br>Estimate: ₹ %{y:,.2f} L<extra></extra>",
                )
            )
        fig.update_layout(
            title=f"Area Sensitivity Curve — {result.location_label} ({bhk} BHK)",
            xaxis_title="Built-up Area (sq. ft.)",
            yaxis_title="Estimated Price (₹ Lakhs)",
            template="plotly_white",
            height=305,
            margin=dict(l=16, r=16, t=44, b=16),
            showlegend=False,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, width="stretch")

        action_col1, action_col2 = st.columns([1.4, 1.0])
        with action_col1:
            comparison_locs = [selected_loc_key] + [k for k in popular_keys if k != selected_loc_key][:4]
            comp_rows = []
            for loc_k in comparison_locs:
                comp_res = predict_price(model, data_columns, loc_k, sqft, bhk, bath)
                comp_rows.append(
                    {
                        "Micro-Market": comp_res.location_label,
                        "Est. Price (₹ L)": comp_res.estimated_lakhs,
                        "₹ / sq. ft.": int(comp_res.price_per_sqft),
                    }
                )
            with st.expander(f"Compare {bhk} BHK ({sqft:,} sq. ft.) across popular hubs", expanded=False):
                st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

        with action_col2:
            report_text = _format_valuation_report(result, locality_bench, city_bench)
            st.download_button(
                label="Download Valuation Brief (.txt)",
                data=report_text,
                file_name=f"brickwise_{selected_loc_key.replace(' ', '_')}_{bhk}bhk.txt",
                mime="text/plain",
                width="stretch",
            )


def _render_market_explorer_tab(market_df: pd.DataFrame, city_bench: dict) -> None:
    f1, f2, f3 = st.columns([1.1, 1.2, 1.2])
    with f1:
        bhk_filter = st.selectbox("Filter by Configuration", ["All", "1 BHK", "2 BHK", "3 BHK", "4+ BHK"])
    with f2:
        max_price = st.slider("Max Listing Price (₹ Lakhs)", 40, 600, 300, 10)
    with f3:
        min_listings = st.slider("Min. Listings per Locality (Ranking)", 10, 80, 25, 5)

    filtered = market_df[market_df["price_lakhs"] <= max_price].copy()
    if bhk_filter == "1 BHK":
        filtered = filtered[filtered["bhk_num"] == 1]
    elif bhk_filter == "2 BHK":
        filtered = filtered[filtered["bhk_num"] == 2]
    elif bhk_filter == "3 BHK":
        filtered = filtered[filtered["bhk_num"] == 3]
    elif bhk_filter == "4+ BHK":
        filtered = filtered[filtered["bhk_num"] >= 4]

    if filtered.empty:
        st.warning("No historical listings match these filters. Broaden the price or BHK range.")
        return

    filtered_bench = get_city_benchmarks(filtered)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching Listings", f"{filtered_bench['listing_count']:,}")
    k2.metric("Median Price", f"₹ {filtered_bench['median_price_lakhs']:,.1f} L")
    k3.metric("Median Rate", f"₹ {filtered_bench['median_ppsqft']:,.0f} / sq.ft")
    k4.metric("Median Area", f"{filtered_bench['median_sqft']:,.0f} sq.ft")

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        top_locs = get_top_localities_summary(filtered, min_listings=min_listings, top_n=15)
        if top_locs.empty:
            st.info("Lower the minimum listings threshold to display top micro-markets.")
        else:
            top_sorted = top_locs.sort_values("median_ppsqft", ascending=True)
            fig_bar = px.bar(
                top_sorted,
                x="median_ppsqft",
                y="location_label",
                orientation="h",
                text="median_ppsqft",
                color="median_ppsqft",
                color_continuous_scale=["#CBD5E1", "#B84A27"],
                title="Top Micro-Markets by Median ₹ / sq. ft.",
                labels={"median_ppsqft": "Median ₹ / sq. ft.", "location_label": ""},
            )
            fig_bar.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
            fig_bar.update_layout(
                coloraxis_showscale=False,
                template="plotly_white",
                height=440,
                margin=dict(l=10, r=40, t=48, b=16),
                xaxis=dict(range=[0, float(top_sorted["median_ppsqft"].max()) * 1.18]),
            )
            st.plotly_chart(fig_bar, width="stretch")

    with chart_right:
        cap = float(filtered["price_lakhs"].quantile(0.98))
        hist_df = filtered[filtered["price_lakhs"] <= cap]
        fig_hist = px.histogram(
            hist_df,
            x="price_lakhs",
            nbins=32,
            title="Price Distribution Across Bengaluru (up to 98th pct.)",
            labels={"price_lakhs": "Listing Price (₹ Lakhs)", "count": "Listings"},
            color_discrete_sequence=["#0F172A"],
        )
        fig_hist.add_vline(
            x=filtered_bench["median_price_lakhs"],
            line_dash="dash",
            line_color="#B84A27",
            annotation_text=f"Median ₹ {filtered_bench['median_price_lakhs']:,.1f}L",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            template="plotly_white",
            height=440,
            margin=dict(l=16, r=16, t=48, b=16),
            bargap=0.06,
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("#### Locality Deep-Dive")
    available_localities = sorted(filtered["location_label"].unique().tolist())
    default_inspect = "Whitefield" if "Whitefield" in available_localities else available_localities[0]
    inspect_label = st.selectbox(
        "Inspect Historical Listings by Locality",
        options=available_localities,
        index=available_localities.index(default_inspect),
    )
    loc_stats = get_locality_benchmark(filtered, inspect_label)
    if loc_stats:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Listings Found", f"{loc_stats['listing_count']}")
        d2.metric("Median Price", f"₹ {loc_stats['median_price_lakhs']:,.1f} L")
        d3.metric("IQR (25th–75th)", f"₹ {loc_stats['p25_price_lakhs']:,.0f}L – ₹ {loc_stats['p75_price_lakhs']:,.0f}L")
        d4.metric("Typical Layout", f"{loc_stats['common_bhk']} BHK")

        sample_df = loc_stats["sample_listings"].copy()
        sample_df["Sq. Ft."] = sample_df["Sq. Ft."].round(0).astype(int)
        sample_df["₹ / Sq. Ft."] = sample_df["₹ / Sq. Ft."].round(0).astype(int)
        st.dataframe(sample_df, width="stretch", hide_index=True)


def _render_methodology_tab(model: object, data_columns: list[str]) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Algorithm", "Linear Regression")
    m2.metric("Holdout Test R²", f"{MODEL_METRICS['r2_score']:.4f}")
    m3.metric("Mean Absolute Error", f"₹ {MODEL_METRICS['mae_lakhs']:.2f} L")
    m4.metric("5-Fold CV Mean R²", f"{MODEL_METRICS['cv_r2_mean']:.4f}")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        bench_df = pd.DataFrame(MODEL_BENCHMARKS)
        fig_bench = px.bar(
            bench_df,
            x="Model",
            y="GridSearch CV R²",
            color="Selected",
            color_discrete_map={"Yes": "#B84A27", "No": "#94A3B8"},
            text="GridSearch CV R²",
            title="Model Selection Benchmark (GridSearchCV)",
        )
        fig_bench.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_bench.update_layout(
            template="plotly_white",
            height=340,
            yaxis=dict(range=[0, 1.0]),
            margin=dict(l=16, r=16, t=48, b=16),
        )
        st.plotly_chart(fig_bench, width="stretch")

    with c2:
        cv_df = pd.DataFrame(
            {
                "Fold": [f"Split {i}" for i in range(1, 6)],
                "R² Score": MODEL_METRICS["cv_r2_scores"],
            }
        )
        fig_cv = px.line(
            cv_df,
            x="Fold",
            y="R² Score",
            markers=True,
            title="5-Fold ShuffleSplit Cross-Validation Stability",
        )
        fig_cv.update_traces(line=dict(color="#0F172A", width=3), marker=dict(size=9, color="#B84A27"))
        fig_cv.update_layout(
            template="plotly_white",
            height=340,
            yaxis=dict(range=[0.65, 0.95]),
            margin=dict(l=16, r=16, t=48, b=16),
        )
        st.plotly_chart(fig_cv, width="stretch")

    coefs = get_locality_coefficients(model, data_columns)
    if coefs:
        coef_df = pd.DataFrame(coefs)
        st.markdown("#### Learned Micro-Market Price Weights")
        p1, p2 = st.columns(2, gap="large")
        with p1:
            st.caption("Top 10 Premium Micro-Markets (Additive Model Effect in ₹ Lakhs)")
            st.dataframe(
                coef_df.head(10)[["locality", "premium_lakhs"]].rename(
                    columns={"locality": "Locality", "premium_lakhs": "Weight (₹ Lakhs)"}
                ),
                width="stretch",
                hide_index=True,
            )
        with p2:
            st.caption("Top 10 Value Micro-Markets (Additive Model Effect in ₹ Lakhs)")
            st.dataframe(
                coef_df.tail(10).sort_values("premium_lakhs")[["locality", "premium_lakhs"]].rename(
                    columns={"locality": "Locality", "premium_lakhs": "Weight (₹ Lakhs)"}
                ),
                width="stretch",
                hide_index=True,
            )

    st.info(
        "Methodology Note: Predictions use an additive Multiple Linear Regression model trained on "
        "13,320 historical Bengaluru listings after range-averaging square footage, removing per-BHK "
        "outliers, and one-hot encoding 241 micro-markets. Estimates should be paired with on-site "
        "inspection, floor-rise, facing, and builder-grade adjustments."
    )


def main() -> None:
    model, data_columns, locations = get_model_resources()
    market_df = get_market_data()
    city_bench = get_city_benchmarks(market_df)

    _render_header(city_bench=city_bench, locations_count=len(locations))

    tab_estimate, tab_market, tab_model = st.tabs(
        [
            "Property Estimator",
            "Bengaluru Market Explorer",
            "Model & Methodology",
        ]
    )

    with tab_estimate:
        _render_estimator_tab(
            model=model,
            data_columns=data_columns,
            locations=locations,
            market_df=market_df,
            city_bench=city_bench,
        )

    with tab_market:
        _render_market_explorer_tab(market_df=market_df, city_bench=city_bench)

    with tab_model:
        _render_methodology_tab(model=model, data_columns=data_columns)


if __name__ == "__main__":
    main()
