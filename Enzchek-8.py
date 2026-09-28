#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:49:37 2026

@author: masonleffler
"""

#!/usr/bin/env python3
"""Streamlit app for plate-reader enzymatic kinetics data."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Plate Reader Enzyme Kinetics", layout="wide")
st.title("Plate Reader Enzyme Kinetics")
st.caption("Upload plate-reader data, subtract replicate-specific blank wells, convert absorbance to product concentration, and determine initial rates.")

# -----------------------------------------------------------------------------
# Core data-processing functions
# -----------------------------------------------------------------------------

def read_plate_data(uploaded_file):
    """Read row 46 headers / row 47 onward data from the uploaded workbook."""
    raw = pd.read_excel(uploaded_file, header=45)
    if raw.shape[1] < 39:
        raise ValueError(
            f"The workbook must contain at least 39 columns (through AM); found {raw.shape[1]}."
        )
    # Excel column B is index 1; D:AM are indices 3:39 (36 columns).
    time = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    absorbance = raw.iloc[:, 3:39].apply(pd.to_numeric, errors="coerce")
    absorbance.columns = [str(c) for c in raw.columns[3:39]]

    valid = time.notna()
    time = time.loc[valid].reset_index(drop=True)
    absorbance = absorbance.loc[valid].reset_index(drop=True)

    if len(time) == 0:
        raise ValueError("No numeric time values were found in column B starting at row 47.")
    if absorbance.shape[1] != 36:
        raise ValueError("Could not read exactly 36 absorbance columns from D:AM.")

    # Keep the original well headers for display/reference.
    well_headers = [str(c) for c in raw.columns[3:39]]
    return time.to_numpy(dtype=float), absorbance.to_numpy(dtype=float), well_headers


def subtract_trial_blanks(absorbance):
    """Subtract series 1 from every series within each experimental trial.

    Trial 1: D:O -> column 0 is the blank.
    Trial 2: P:AA -> column 12 is the blank.
    Trial 3: AB:AM -> column 24 is the blank.
    """
    if absorbance.shape[1] != 36:
        raise ValueError("Expected 36 absorbance columns (D:AM).")

    corrected = np.full_like(absorbance, np.nan, dtype=float)
    for trial in range(3):
        start = trial * 12
        stop = start + 12
        block = absorbance[:, start:stop]
        blank = block[:, [0]]
        corrected[:, start:stop] = block - blank
    return corrected


def absorbance_to_product(delta_absorbance, calibration_slope, calibration_intercept, stoich_factor):
    """Convert delta absorbance to product concentration.

    Calibration is defined as:
        A = slope * analyte_concentration + intercept

    For a delta-A trace, the intercept cancels. stoich_factor is the number of
    analyte concentration units produced per product concentration unit.
    Thus product = ((delta_A) / slope) / stoich_factor.
    """
    if calibration_slope == 0:
        raise ValueError("Calibration slope cannot be zero.")
    if stoich_factor <= 0:
        raise ValueError("Stoichiometric factor must be greater than zero.")
    # Keep intercept as an explicit parameter so the calibration definition is clear.
    _ = calibration_intercept
    return (delta_absorbance / calibration_slope) / stoich_factor


def linear_fit(x, y):
    """Return slope, intercept, and R² for a finite x/y subset."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def detect_initial_slope(time, product, min_points=5, max_points=None, min_r2=0.0, require_positive=True):
    """Automatically select the best initial contiguous window starting at t=0.

    Every candidate window starts at the first valid time point. Among windows
    meeting the constraints, choose the one with highest R²; ties favor the
    shorter window. This is intentionally constrained to the initial region so
    later nonlinear behavior cannot be selected as the 'initial' slope.
    """
    time = np.asarray(time, dtype=float)
    product = np.asarray(product, dtype=float)
    valid = np.isfinite(time) & np.isfinite(product)
    t = time[valid]
    y = product[valid]

    if len(t) < min_points:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "start": np.nan, "end": np.nan, "n": 0}

    max_n = len(t) if max_points is None else min(int(max_points), len(t))
    min_n = min(int(min_points), max_n)
    candidates = []

    for n in range(min_n, max_n + 1):
        slope, intercept, r2 = linear_fit(t[:n], y[:n])
        if not np.isfinite(slope) or not np.isfinite(r2):
            continue
        if r2 + 1e-12 < min_r2:
            continue
        if require_positive and slope <= 0:
            continue
        candidates.append((r2, -n, slope, intercept, n))

    if not candidates:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "start": np.nan, "end": np.nan, "n": 0}

    # Highest R²; for a tie, shorter initial window.
    best = max(candidates, key=lambda z: (z[0], z[1]))
    r2, neg_n, slope, intercept, n = best
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "start": float(t[0]),
        "end": float(t[n - 1]),
        "n": int(n),
    }


def calculate_rates(time, product, mode, min_points, max_points, min_r2, require_positive):
    """Calculate initial rate/R² for all 36 traces."""
    results = []
    for i in range(product.shape[1]):
        if mode == "Automatic":
            result = detect_initial_slope(
                time, product[:, i], min_points=min_points,
                max_points=max_points, min_r2=min_r2,
                require_positive=require_positive,
            )
        else:
            n = min(max_points, len(time))
            slope, intercept, r2 = linear_fit(time[:n], product[:n, i])
            result = {"slope": slope, "intercept": intercept, "r2": r2,
                      "start": float(time[0]), "end": float(time[n - 1]), "n": n}
        results.append(result)
    return results


def build_rate_table(results, concentrations):
    """Arrange rates as substrate concentration + three experimental trials."""
    rows = []

    for series in range(12):
        row = {
            "Substrate concentration": concentrations[series],
            "Trial 1": results[series]["slope"],
            "Trial 2": results[12 + series]["slope"],
            "Trial 3": results[24 + series]["slope"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def make_plot(time, product, concentrations, trial, title, y_title, font_size, title_size,
              x_min, x_max, y_min, y_max, legend_position, line_width):
    fig = go.Figure()
    start = trial * 12
    for series in range(12):
        label = f"{concentrations[series]:g}"
        fig.add_trace(go.Scatter(
            x=time,
            y=product[:, start + series],
            mode="lines",
            name=label,
            line={"width": line_width},
        ))
    fig.update_layout(
        title={"text": title, "font": {"size": title_size}},
        xaxis={"title": "Time", "range": [x_min, x_max] if x_min < x_max else None,
               "title_font": {"size": font_size}, "tickfont": {"size": font_size}},
        yaxis={"title": y_title, "range": [y_min, y_max] if y_min < y_max else None,
               "title_font": {"size": font_size}, "tickfont": {"size": font_size}},
        legend={"x": 1.02, "y": 1 if legend_position == "Right" else 0.5,
                "orientation": "v" if legend_position == "Right" else "h",
                "font": {"size": font_size}},
        margin={"l": 70, "r": 160 if legend_position == "Right" else 40, "t": 70, "b": 60},
        font={"size": font_size},
        hovermode="x unified",
    )
    return fig


# -----------------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------------
st.subheader("1. Upload plate-reader workbook")
uploaded_file = st.file_uploader("Upload .xlsx file", type=["xlsx"])

if uploaded_file is None:
    st.info("Upload an .xlsx workbook to begin. The app expects headers on row 46, time in column B, and absorbance in D:AM.")
    st.stop()

try:
    time, raw_abs, well_headers = read_plate_data(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the workbook: {exc}")
    st.stop()

st.success(f"Loaded {len(time)} time points and 36 absorbance traces from columns D:AM.")

# -----------------------------------------------------------------------------
# Substrate concentrations
# -----------------------------------------------------------------------------
st.subheader("2. Enter substrate concentrations")
st.write("Enter one concentration for each of the 12 series. The same 12 concentrations are used for all three experimental replicates.")

conc_cols = st.columns(4)
concentrations = []
for i in range(12):
    with conc_cols[i % 4]:
        concentrations.append(
            st.number_input(
                f"Series {i + 1}", value=0.0, step=1.0, format="%.6g",
                key=f"conc_{i}", help=f"Substrate concentration for series {i + 1}."
            )
        )

# -----------------------------------------------------------------------------
# Blank subtraction and calibration
# -----------------------------------------------------------------------------
st.subheader("3. Baseline subtraction and calibration")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Baseline subtraction**")
    st.caption("Within each replicate, series 1 is subtracted from all 12 series at the same time point.")
    st.code("Trial 1: D:O − D\nTrial 2: P:AA − P\nTrial 3: AB:AM − AB")
with col2:
    cal_slope = st.number_input("Calibration slope", value=0.00192, format="%.10g", help="Slope m in A = m·concentration + b.")
    cal_intercept = st.number_input("Calibration intercept", value=0.0012, format="%.10g", help="Intercept b in A = m·concentration + b. Delta-A processing cancels this term.")
with col3:
    stoich_factor = st.number_input(
        "Analyte equivalents per product",
        min_value=0.000001, value=2.0, step=1.0, format="%.6g",
        help="For example, enter 2 if two phosphate molecules correspond to one product molecule. Product concentration = calibrated analyte concentration / this value."
    )

try:
    delta_abs = subtract_trial_blanks(raw_abs)
    product = absorbance_to_product(delta_abs, cal_slope, cal_intercept, stoich_factor)
except Exception as exc:
    st.error(str(exc))
    st.stop()

# -----------------------------------------------------------------------------
# Plot customization
# -----------------------------------------------------------------------------
st.subheader("4. Product concentration vs. time")
plot_a, plot_b, plot_c = st.columns(3)
with plot_a:
    x_auto = st.checkbox("Automatic x-axis", value=True)
    y_auto = st.checkbox("Automatic y-axis", value=True)
    font_size = st.slider("Axis/legend font size", 8, 28, 14)
with plot_b:
    title_size = st.slider("Title font size", 10, 36, 18)
    line_width = st.slider("Line width", 1.0, 6.0, 2.0, 0.5)
    legend_position = st.selectbox("Legend", ["Right", "Bottom"])
with plot_c:
    if x_auto:
        x_min, x_max = float(np.nanmin(time)), float(np.nanmax(time))
    else:
        x_min = st.number_input("X minimum", value=float(np.nanmin(time)))
        x_max = st.number_input("X maximum", value=float(np.nanmax(time)))
    if y_auto:
        finite_product = product[np.isfinite(product)]
        y_min, y_max = (float(np.min(finite_product)), float(np.max(finite_product))) if len(finite_product) else (0.0, 1.0)
    else:
        y_min = st.number_input("Y minimum", value=0.0)
        y_max = st.number_input("Y maximum", value=1.0)

y_title = st.text_input("Y-axis title", "Product concentration")

figures = []
for trial in range(3):
    figures.append(make_plot(
        time, product, concentrations, trial,
        title=f"Trial {trial + 1}", y_title=y_title,
        font_size=font_size, title_size=title_size,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        legend_position=legend_position, line_width=line_width,
    ))

for i, fig in enumerate(figures, start=1):
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")

# -----------------------------------------------------------------------------
# Initial slope detection
# -----------------------------------------------------------------------------
st.subheader("5. Initial-rate determination")
mode = st.radio("Slope selection", ["Automatic", "First N points"], horizontal=True)
sl1, sl2, sl3, sl4 = st.columns(4)
with sl1:
    min_points = st.number_input("Minimum points", min_value=2, max_value=max(2, len(time)), value=min(5, len(time)), step=1)
with sl2:
    max_points = st.number_input("Maximum points", min_value=2, max_value=max(2, len(time)), value=min(15, len(time)), step=1)
with sl3:
    min_r2 = st.number_input("Minimum R²", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
with sl4:
    require_positive = st.checkbox("Require positive slope", value=True)

if min_points > max_points:
    st.error("Minimum points cannot exceed maximum points.")
    st.stop()

results = calculate_rates(time, product, mode, int(min_points), int(max_points), float(min_r2), require_positive)

rate_rows = []
for trial in range(3):
    for series in range(12):
        r = results[trial * 12 + series]
        rate_rows.append({
            "Trial": trial + 1,
            "Series": series + 1,
            "Substrate concentration": concentrations[series],
            "Initial rate": r["slope"],
            "R²": r["r2"],
            "Points used": r["n"],
            "Fit start": r["start"],
            "Fit end": r["end"],
        })
rate_details = pd.DataFrame(rate_rows)
st.dataframe(rate_details, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# CSV output
# -----------------------------------------------------------------------------
st.subheader("6. Export initial rates")
output = build_rate_table(results, concentrations)
output_csv = output.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download initial-rates CSV",
    data=output_csv,
    file_name="initial_rates.csv",
    mime="text/csv",
)
st.dataframe(output, use_container_width=True, hide_index=True)
