import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================================================
# Jakarta Fire-Air Quality Scenario Prediction Dashboard
# GA2 Data Product Version 3 with pollution severity gauges
#
# Main purpose:
#   Train models using 2023 Jakarta air-quality and Indonesian fire data,
#   then allow users to input a current/hypothetical fire scenario and
#   predict expected PM2.5 and PM10 values.
# =========================================================

YEAR = 2023
RANDOM_STATE = 42

AIR_PATH = "ispu_dki_all.csv"
FIRE_PATH = "fire_archive_SV-C2_744546.csv"
PROCESSED_PATH = os.path.join("processed_data", "model_dataset_2023.csv")

POLLUTANTS = ["pm25", "pm10", "so2", "co", "o3", "no2"]
FIRE_COLS = [
    "fire_count", "total_frp", "avg_frp", "max_frp",
    "low_fire_count", "medium_fire_count", "high_fire_count"
]

FEATURE_COLS = [
    "fire_count", "total_frp", "avg_frp", "max_frp",
    "low_fire_count", "medium_fire_count", "high_fire_count",

    "fire_count_lag1", "fire_count_lag2", "fire_count_lag3", "fire_count_lag7",
    "total_frp_lag1", "total_frp_lag2", "total_frp_lag3", "total_frp_lag7",
    "high_fire_count_lag1", "high_fire_count_lag2", "high_fire_count_lag3", "high_fire_count_lag7",

    "fire_count_roll3", "fire_count_roll7", "fire_count_roll14",
    "total_frp_roll3", "total_frp_roll7", "total_frp_roll14",
    "high_fire_count_roll3", "high_fire_count_roll7", "high_fire_count_roll14",

    "month", "dayofyear", "fire_season"
]

MODEL_DEFINITIONS = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        ))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ))
    ])
}

# Color scale for pollution communication.
# The dashboard values are model outputs from the project dataset, not official AQI conversions.
# These thresholds are used as an intuitive display scale for decision-support storytelling.
POLLUTION_LEVELS = [
    {"label": "Good", "low": 0, "high": 50, "color": "#00E400", "text_color": "#111111"},
    {"label": "Moderate", "low": 50, "high": 100, "color": "#FFFF00", "text_color": "#111111"},
    {"label": "Unhealthy for Sensitive Groups", "low": 100, "high": 150, "color": "#FF7E00", "text_color": "#111111"},
    {"label": "Unhealthy", "low": 150, "high": 200, "color": "#FF0000", "text_color": "#FFFFFF"},
    {"label": "Very Unhealthy", "low": 200, "high": 300, "color": "#8F3F97", "text_color": "#FFFFFF"},
    {"label": "Hazardous", "low": 300, "high": 500, "color": "#7E0023", "text_color": "#FFFFFF"},
]


def pollution_level(value):
    if pd.isna(value):
        return {"label": "Not available", "color": "#D9D9D9", "text_color": "#111111"}
    value = max(float(value), 0)
    for level in POLLUTION_LEVELS:
        if value <= level["high"]:
            return level
    return POLLUTION_LEVELS[-1]


def pm25_risk_category(value):
    # Kept for compatibility with the report wording.
    return pollution_level(value)["label"]


def risk_message(category):
    messages = {
        "Good": "Predicted PM2.5 is Good. Routine monitoring is sufficient.",
        "Moderate": "Predicted PM2.5 is Moderate. Sensitive groups should pay attention.",
        "Unhealthy for Sensitive Groups": "Predicted PM2.5 may affect sensitive groups. Prepare targeted advisory messages.",
        "Unhealthy": "Predicted PM2.5 is Unhealthy. Increase monitoring and prepare public health advisory.",
        "Very Unhealthy": "Predicted PM2.5 is Very Unhealthy. Strengthen warnings and consider reducing outdoor exposure.",
        "Hazardous": "Predicted PM2.5 is Hazardous. Issue urgent public health communication and coordinate response actions.",
        "Not available": "PM2.5 prediction is not available."
    }
    return messages.get(category, "Risk category unavailable.")


def suggested_action(category):
    if category == "Good":
        return "Routine monitoring. No special public warning is required."
    if category == "Moderate":
        return "Continue monitoring and prepare advisory messages for highly sensitive groups."
    if category == "Unhealthy for Sensitive Groups":
        return "Increase attention to sensitive groups such as children, elderly people, outdoor workers, and people with respiratory disease."
    if category == "Unhealthy":
        return "Increase monitoring frequency, prepare public health advisory, and coordinate with related agencies."
    if category == "Very Unhealthy":
        return "Strengthen public warnings, reduce prolonged outdoor activity, and coordinate cross-agency response."
    if category == "Hazardous":
        return "Urgent public health communication is needed. Consider emergency response coordination and strong exposure-reduction advice."
    return "Prediction is not available."


def pollutant_card(label, value):
    level = pollution_level(value)
    color = level["color"]
    text_color = level["text_color"]
    level_label = level["label"]
    return f"""
    <div style="background-color:{color}; color:{text_color};
                padding:18px; border-radius:16px; text-align:center;
                box-shadow:0 2px 8px rgba(0,0,0,0.12); min-height:145px;">
        <div style="font-size:18px; font-weight:700; margin-bottom:8px;">{label}</div>
        <div style="font-size:42px; font-weight:800; line-height:1;">{value:.2f}</div>
        <div style="font-size:17px; font-weight:700; margin-top:10px;">{level_label}</div>
    </div>
    """


def plot_pollution_gauge(value, title, max_value=300):
    fig, ax = plt.subplots(figsize=(6.2, 3.4), subplot_kw={"aspect": "equal"})
    capped_value = min(max(float(value), 0), max_value)

    for level in POLLUTION_LEVELS:
        lo = max(level["low"], 0)
        hi = min(level["high"], max_value)
        if lo >= max_value:
            continue
        theta1 = 180 - (hi / max_value) * 180
        theta2 = 180 - (lo / max_value) * 180
        wedge = plt.matplotlib.patches.Wedge(
            (0, 0), 1.0, theta1, theta2, width=0.25,
            facecolor=level["color"], edgecolor="white", linewidth=2
        )
        ax.add_patch(wedge)

    angle = np.deg2rad(180 - (capped_value / max_value) * 180)
    needle_x = 0.78 * np.cos(angle)
    needle_y = 0.78 * np.sin(angle)
    ax.plot([0, needle_x], [0, needle_y], linewidth=3, color="#222222")
    ax.scatter([0], [0], s=80, color="#222222", zorder=5)

    level = pollution_level(value)
    ax.text(0, -0.18, f"{value:.2f}", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.35, level["label"], ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0, 1.08, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(-1.0, -0.08, "0", ha="center", va="center", fontsize=9)
    ax.text(1.0, -0.08, str(max_value), ha="center", va="center", fontsize=9)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.45, 1.15)
    ax.axis("off")
    fig.tight_layout()
    return fig


def color_legend_html():
    blocks = []
    for level in POLLUTION_LEVELS[:-1]:
        display_low = int(level["low"]) + 1 if level["low"] else 0
        block = (
            f'<span style="display:inline-block; background-color:{level["color"]}; color:{level["text_color"]}; '
            f'padding:6px 10px; border-radius:10px; margin:3px; font-size:13px; font-weight:600;">'
            f'{level["label"]}<br>{display_low}-{int(level["high"])}<span style="display:none"></span></span>'
        )
        blocks.append(block)
    last = POLLUTION_LEVELS[-1]
    blocks.append(
        f'<span style="display:inline-block; background-color:{last["color"]}; color:{last["text_color"]}; '
        f'padding:6px 10px; border-radius:10px; margin:3px; font-size:13px; font-weight:600;">'
        f'Hazardous<br>301+</span>'
    )
    return "".join(blocks)



@st.cache_data(show_spinner=False)
def build_dataset_from_raw():
    """Build the processed modelling dataset if raw CSV files are available."""
    if not (os.path.exists(AIR_PATH) and os.path.exists(FIRE_PATH)):
        return None

    air = pd.read_csv(AIR_PATH)
    fire = pd.read_csv(FIRE_PATH)

    # Air-quality cleaning
    air["date"] = pd.to_datetime(air["tanggal"], errors="coerce")
    air = air.dropna(subset=["date"])
    air_2023 = air[air["date"].dt.year == YEAR].copy()
    air_2023 = air_2023.dropna(subset=["stasiun"])

    for col in POLLUTANTS:
        air_2023[col] = pd.to_numeric(air_2023[col], errors="coerce")

    air_2023 = air_2023.drop_duplicates(subset=["date", "stasiun"])
    air_2023 = air_2023.sort_values(["stasiun", "date"])

    low_missing_pollutants = ["pm10", "so2", "co", "o3", "no2"]
    for col in low_missing_pollutants:
        air_2023[col] = air_2023.groupby("stasiun")[col].transform(
            lambda x: x.interpolate(method="linear", limit=3, limit_direction="both")
        )

    daily_air = (
        air_2023.groupby("date", as_index=False)[POLLUTANTS]
        .mean()
        .sort_values("date")
    )

    # Fire hotspot cleaning
    fire["date"] = pd.to_datetime(fire["acq_date"], errors="coerce")
    fire = fire.dropna(subset=["date", "latitude", "longitude"])
    fire_2023 = fire[fire["date"].dt.year == YEAR].copy()

    for col in ["latitude", "longitude", "frp"]:
        fire_2023[col] = pd.to_numeric(fire_2023[col], errors="coerce")

    fire_2023 = fire_2023.dropna(subset=["latitude", "longitude", "frp"])
    fire_2023 = fire_2023.drop_duplicates()

    # Spatial filtering: Sumatra and Kalimantan
    sumatra = (
        (fire_2023["latitude"].between(-6, 6)) &
        (fire_2023["longitude"].between(95, 106))
    )
    kalimantan = (
        (fire_2023["latitude"].between(-8, 2)) &
        (fire_2023["longitude"].between(109, 120))
    )
    fire_core = fire_2023[sumatra | kalimantan].copy()

    fire_core["low_fire"] = (fire_core["frp"] <= 5).astype(int)
    fire_core["medium_fire"] = ((fire_core["frp"] > 5) & (fire_core["frp"] <= 10)).astype(int)
    fire_core["high_fire"] = (fire_core["frp"] > 10).astype(int)

    daily_fire = (
        fire_core.groupby("date")
        .agg(
            fire_count=("latitude", "count"),
            total_frp=("frp", "sum"),
            avg_frp=("frp", "mean"),
            max_frp=("frp", "max"),
            low_fire_count=("low_fire", "sum"),
            medium_fire_count=("medium_fire", "sum"),
            high_fire_count=("high_fire", "sum")
        )
        .reset_index()
        .sort_values("date")
    )

    calendar = pd.DataFrame({
        "date": pd.date_range(start=f"{YEAR}-01-01", end=f"{YEAR}-12-31", freq="D")
    })
    df = calendar.merge(daily_air, on="date", how="left")
    df = df.merge(daily_fire, on="date", how="left")
    df[FIRE_COLS] = df[FIRE_COLS].fillna(0)

    df = add_missing_features(df)

    os.makedirs("processed_data", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return df


def add_missing_features(df):
    """Ensure the modelling dataset contains all engineered features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Support alternative naming from earlier exploratory notebooks.
    if "mean_frp" in df.columns and "avg_frp" not in df.columns:
        df["avg_frp"] = df["mean_frp"]

    for col in FIRE_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "dayofyear" not in df.columns:
        df["dayofyear"] = df["date"].dt.dayofyear
    if "fire_season" not in df.columns:
        df["fire_season"] = df["month"].isin([8, 9, 10]).astype(int)

    for lag in [1, 2, 3, 7]:
        if f"fire_count_lag{lag}" not in df.columns:
            df[f"fire_count_lag{lag}"] = df["fire_count"].shift(lag).fillna(0)
        if f"total_frp_lag{lag}" not in df.columns:
            df[f"total_frp_lag{lag}"] = df["total_frp"].shift(lag).fillna(0)
        if f"high_fire_count_lag{lag}" not in df.columns:
            df[f"high_fire_count_lag{lag}"] = df["high_fire_count"].shift(lag).fillna(0)

    for window in [3, 7, 14]:
        if f"fire_count_roll{window}" not in df.columns:
            df[f"fire_count_roll{window}"] = df["fire_count"].rolling(window, min_periods=1).mean()
        if f"total_frp_roll{window}" not in df.columns:
            df[f"total_frp_roll{window}"] = df["total_frp"].rolling(window, min_periods=1).mean()
        if f"high_fire_count_roll{window}" not in df.columns:
            df[f"high_fire_count_roll{window}"] = df["high_fire_count"].rolling(window, min_periods=1).mean()

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_dataset():
    possible_paths = [PROCESSED_PATH, "model_dataset_2023.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return add_missing_features(df)
    return build_dataset_from_raw()


@st.cache_data(show_spinner=False)
def evaluate_models(df, target_col):
    model_data = df.dropna(subset=[target_col]).copy()
    X = model_data[FEATURE_COLS]
    y = model_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []
    preds = {}

    for model_name, model_template in MODEL_DEFINITIONS.items():
        fitted = clone(model_template).fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        results.append({
            "target": target_col,
            "model": model_name,
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "train_rows": len(X_train),
            "test_rows": len(X_test)
        })
        preds[model_name] = pd.DataFrame({
            "actual": y_test.values,
            "predicted": y_pred
        })

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    return results_df, preds


@st.cache_resource(show_spinner=False)
def fit_full_model(df, target_col, model_name):
    model_data = df.dropna(subset=[target_col]).copy()
    X = model_data[FEATURE_COLS]
    y = model_data[target_col]
    model = clone(MODEL_DEFINITIONS[model_name])
    model.fit(X, y)
    return model


def get_rf_feature_importance(df, target_col):
    model = fit_full_model(df, target_col, "Random Forest")
    rf = model.named_steps["model"]
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    return imp


def plot_actual_vs_predicted(pred_df, target_label, model_name):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.7)
    min_v = min(pred_df["actual"].min(), pred_df["predicted"].min())
    max_v = max(pred_df["actual"].max(), pred_df["predicted"].max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(f"Actual vs Predicted ({model_name})")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df, target_label):
    top = importance_df.head(10).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 10 Random Forest Feature Importance for {target_label}")
    fig.tight_layout()
    return fig


def historical_percentile_defaults(df):
    """Create reasonable default values using the training dataset."""
    fire_season_df = df[df["fire_season"] == 1].copy()
    base = fire_season_df if not fire_season_df.empty else df

    defaults = {}
    for col in FIRE_COLS:
        defaults[col] = float(base[col].quantile(0.75)) if col in base.columns else 0.0
    defaults["month"] = 9
    defaults["dayofyear"] = 260
    defaults["fire_season"] = 1
    return defaults


def build_scenario_input(values, mode="Simple"):
    """Convert user inputs into one row with exactly the required modelling features."""
    row = {
        "fire_count": values["fire_count"],
        "total_frp": values["total_frp"],
        "avg_frp": values["avg_frp"],
        "max_frp": values["max_frp"],
        "low_fire_count": values["low_fire_count"],
        "medium_fire_count": values["medium_fire_count"],
        "high_fire_count": values["high_fire_count"],
        "month": values["month"],
        "dayofyear": values["dayofyear"],
        "fire_season": values["fire_season"],
    }

    if mode == "Advanced":
        for col in [
            "fire_count_lag1", "fire_count_lag2", "fire_count_lag3", "fire_count_lag7",
            "total_frp_lag1", "total_frp_lag2", "total_frp_lag3", "total_frp_lag7",
            "high_fire_count_lag1", "high_fire_count_lag2", "high_fire_count_lag3", "high_fire_count_lag7",
            "fire_count_roll3", "fire_count_roll7", "fire_count_roll14",
            "total_frp_roll3", "total_frp_roll7", "total_frp_roll14",
            "high_fire_count_roll3", "high_fire_count_roll7", "high_fire_count_roll14",
        ]:
            row[col] = values[col]
    else:
        # Simple mode: use the current input as a proxy for recent lag/rolling values.
        # This makes the prototype easy for non-technical users while keeping the
        # feature structure consistent with the 31-predictor training model.
        for lag in [1, 2, 3, 7]:
            row[f"fire_count_lag{lag}"] = values["fire_count"]
            row[f"total_frp_lag{lag}"] = values["total_frp"]
            row[f"high_fire_count_lag{lag}"] = values["high_fire_count"]
        for window in [3, 7, 14]:
            row[f"fire_count_roll{window}"] = values["fire_count"]
            row[f"total_frp_roll{window}"] = values["total_frp"]
            row[f"high_fire_count_roll{window}"] = values["high_fire_count"]

    return pd.DataFrame([row])[FEATURE_COLS]


# ========================= Streamlit UI =========================
st.set_page_config(
    page_title="Jakarta Fire-Air Quality Scenario Prediction Dashboard",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Jakarta Fire-Air Quality Scenario Prediction Dashboard")
st.caption(
    "This data product trains regression models using the 2023 Jakarta air-quality and Indonesian fire dataset. "
    "Users can input current or hypothetical fire indicators to predict expected PM2.5 and PM10 levels. "
    "The output uses an AQI-style color scale to make pollution severity easier to communicate."
)

with st.sidebar:
    st.header("Dashboard Controls")
    st.write("GA2 data product: scenario-based PM2.5 and PM10 prediction")

    df = load_dataset()

    if df is None:
        st.error(
            "Dataset not found. Please run ga2.ipynb first, or place either "
            "processed_data/model_dataset_2023.csv or the raw CSV files in this folder."
        )
        st.stop()

    model_choice = st.selectbox(
        "Select prediction model",
        list(MODEL_DEFINITIONS.keys()),
        index=list(MODEL_DEFINITIONS.keys()).index("Random Forest")
    )

    input_mode = st.radio(
        "Input mode",
        ["Simple", "Advanced"],
        help="Simple mode only asks for main fire indicators. Advanced mode allows lag and rolling features."
    )

    show_historical = st.checkbox("Show historical training-data explorer", value=True)

# Train models for the selected model type.
pm25_model = fit_full_model(df, "pm25", model_choice)
pm10_model = fit_full_model(df, "pm10", model_choice)

defaults = historical_percentile_defaults(df)

# ========================= Section 1: Scenario prediction =========================
st.subheader("1. Scenario Prediction Tool")
st.write(
    "Enter current or hypothetical Indonesian fire-activity indicators. The selected model will estimate "
    "Jakarta PM2.5 and PM10. The PM2.5 prediction is also translated into a risk category."
)

with st.form("scenario_form"):
    st.markdown("**Main fire indicators**")
    c1, c2, c3, c4 = st.columns(4)
    fire_count = c1.number_input("Fire Count", min_value=0.0, value=round(defaults["fire_count"], 2), step=10.0)
    total_frp = c2.number_input("Total FRP", min_value=0.0, value=round(defaults["total_frp"], 2), step=100.0)
    avg_frp = c3.number_input("Average FRP", min_value=0.0, value=round(defaults["avg_frp"], 2), step=1.0)
    max_frp = c4.number_input("Maximum FRP", min_value=0.0, value=round(defaults["max_frp"], 2), step=5.0)

    c5, c6, c7 = st.columns(3)
    low_fire_count = c5.number_input("Low Fire Count", min_value=0.0, value=round(defaults["low_fire_count"], 2), step=10.0)
    medium_fire_count = c6.number_input("Medium Fire Count", min_value=0.0, value=round(defaults["medium_fire_count"], 2), step=10.0)
    high_fire_count = c7.number_input("High Fire Count", min_value=0.0, value=round(defaults["high_fire_count"], 2), step=10.0)

    st.markdown("**Seasonal information**")
    c8, c9, c10 = st.columns(3)
    month = c8.slider("Month", 1, 12, int(defaults["month"]))
    dayofyear = c9.slider("Day of Year", 1, 366, int(defaults["dayofyear"]))
    fire_season_text = c10.selectbox("Fire Season", ["Yes", "No"], index=0)
    fire_season = 1 if fire_season_text == "Yes" else 0

    values = {
        "fire_count": fire_count,
        "total_frp": total_frp,
        "avg_frp": avg_frp,
        "max_frp": max_frp,
        "low_fire_count": low_fire_count,
        "medium_fire_count": medium_fire_count,
        "high_fire_count": high_fire_count,
        "month": month,
        "dayofyear": dayofyear,
        "fire_season": fire_season,
    }

    if input_mode == "Advanced":
        st.markdown("**Lag features: previous fire activity**")
        st.caption("Use these fields when recent fire activity is available. Otherwise, use Simple mode.")
        lag_tabs = st.tabs(["Fire Count Lags", "Total FRP Lags", "High Fire Count Lags"])
        with lag_tabs[0]:
            l1, l2, l3, l7 = st.columns(4)
            values["fire_count_lag1"] = l1.number_input("Fire Count Lag 1", min_value=0.0, value=fire_count, step=10.0)
            values["fire_count_lag2"] = l2.number_input("Fire Count Lag 2", min_value=0.0, value=fire_count, step=10.0)
            values["fire_count_lag3"] = l3.number_input("Fire Count Lag 3", min_value=0.0, value=fire_count, step=10.0)
            values["fire_count_lag7"] = l7.number_input("Fire Count Lag 7", min_value=0.0, value=fire_count, step=10.0)
        with lag_tabs[1]:
            l1, l2, l3, l7 = st.columns(4)
            values["total_frp_lag1"] = l1.number_input("Total FRP Lag 1", min_value=0.0, value=total_frp, step=100.0)
            values["total_frp_lag2"] = l2.number_input("Total FRP Lag 2", min_value=0.0, value=total_frp, step=100.0)
            values["total_frp_lag3"] = l3.number_input("Total FRP Lag 3", min_value=0.0, value=total_frp, step=100.0)
            values["total_frp_lag7"] = l7.number_input("Total FRP Lag 7", min_value=0.0, value=total_frp, step=100.0)
        with lag_tabs[2]:
            l1, l2, l3, l7 = st.columns(4)
            values["high_fire_count_lag1"] = l1.number_input("High Fire Count Lag 1", min_value=0.0, value=high_fire_count, step=10.0)
            values["high_fire_count_lag2"] = l2.number_input("High Fire Count Lag 2", min_value=0.0, value=high_fire_count, step=10.0)
            values["high_fire_count_lag3"] = l3.number_input("High Fire Count Lag 3", min_value=0.0, value=high_fire_count, step=10.0)
            values["high_fire_count_lag7"] = l7.number_input("High Fire Count Lag 7", min_value=0.0, value=high_fire_count, step=10.0)

        st.markdown("**Rolling features: cumulative recent fire activity**")
        r_tabs = st.tabs(["Fire Count Rolling", "Total FRP Rolling", "High Fire Count Rolling"])
        with r_tabs[0]:
            r3, r7, r14 = st.columns(3)
            values["fire_count_roll3"] = r3.number_input("Fire Count Rolling 3-Day", min_value=0.0, value=fire_count, step=10.0)
            values["fire_count_roll7"] = r7.number_input("Fire Count Rolling 7-Day", min_value=0.0, value=fire_count, step=10.0)
            values["fire_count_roll14"] = r14.number_input("Fire Count Rolling 14-Day", min_value=0.0, value=fire_count, step=10.0)
        with r_tabs[1]:
            r3, r7, r14 = st.columns(3)
            values["total_frp_roll3"] = r3.number_input("Total FRP Rolling 3-Day", min_value=0.0, value=total_frp, step=100.0)
            values["total_frp_roll7"] = r7.number_input("Total FRP Rolling 7-Day", min_value=0.0, value=total_frp, step=100.0)
            values["total_frp_roll14"] = r14.number_input("Total FRP Rolling 14-Day", min_value=0.0, value=total_frp, step=100.0)
        with r_tabs[2]:
            r3, r7, r14 = st.columns(3)
            values["high_fire_count_roll3"] = r3.number_input("High Fire Count Rolling 3-Day", min_value=0.0, value=high_fire_count, step=10.0)
            values["high_fire_count_roll7"] = r7.number_input("High Fire Count Rolling 7-Day", min_value=0.0, value=high_fire_count, step=10.0)
            values["high_fire_count_roll14"] = r14.number_input("High Fire Count Rolling 14-Day", min_value=0.0, value=high_fire_count, step=10.0)

    predict_clicked = st.form_submit_button("Predict PM2.5 and PM10")

scenario_input = build_scenario_input(values, input_mode)
pred_pm25 = float(pm25_model.predict(scenario_input)[0])
pred_pm10 = float(pm10_model.predict(scenario_input)[0])
risk = pm25_risk_category(pred_pm25)

st.markdown("### Prediction Output")
level_pm25 = pollution_level(pred_pm25)
level_pm10 = pollution_level(pred_pm10)

summary_cols = st.columns(3)
summary_cols[0].metric("Selected Model", model_choice)
summary_cols[1].metric("PM2.5 Severity", level_pm25["label"])
summary_cols[2].metric("PM10 Severity", level_pm10["label"])

card1, card2 = st.columns(2)
with card1:
    st.markdown(pollutant_card("Predicted PM2.5", pred_pm25), unsafe_allow_html=True)
with card2:
    st.markdown(pollutant_card("Predicted PM10", pred_pm10), unsafe_allow_html=True)

st.markdown("#### Pollution Severity Gauges")
st.caption(
    "The colors follow a common AQI-style communication pattern: green means good, yellow means moderate, "
    "orange/red means unhealthy, and purple/maroon means very serious pollution. "
    "In this project, the scale is used as a visual decision-support guide for predicted PM2.5 and PM10 values."
)
st.markdown(color_legend_html(), unsafe_allow_html=True)

gauge1, gauge2 = st.columns(2)
with gauge1:
    st.pyplot(plot_pollution_gauge(pred_pm25, "PM2.5 Pollution Severity"))
with gauge2:
    st.pyplot(plot_pollution_gauge(pred_pm10, "PM10 Pollution Severity"))

if risk in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
    st.error(risk_message(risk))
elif risk in ["Moderate", "Unhealthy for Sensitive Groups"]:
    st.warning(risk_message(risk))
else:
    st.success(risk_message(risk))

st.write("**Suggested administrative action based on PM2.5:**", suggested_action(risk))

with st.expander("Show model input row used for prediction"):
    st.dataframe(scenario_input, use_container_width=True)

st.info(
    "Prototype note: Simple mode automatically uses the current fire indicators as proxies for lag and rolling features. "
    "For a more realistic operational use case, choose Advanced mode and enter recent historical fire activity values."
)

# ========================= Section 2: Model performance =========================
st.subheader("2. Model Performance on 2023 Training Data")
st.write("This section explains how the candidate models performed during GA2 modelling.")

pm25_results, pm25_preds = evaluate_models(df, "pm25")
pm10_results, pm10_preds = evaluate_models(df, "pm10")

p1, p2 = st.columns(2)
with p1:
    st.write("**PM2.5 Model Evaluation**")
    st.dataframe(
        pm25_results.assign(
            R2=pm25_results["R2"].round(3),
            MAE=pm25_results["MAE"].round(2),
            RMSE=pm25_results["RMSE"].round(2)
        ),
        use_container_width=True
    )
    st.bar_chart(pm25_results.set_index("model")[["R2"]])
with p2:
    st.write("**PM10 Model Evaluation**")
    st.dataframe(
        pm10_results.assign(
            R2=pm10_results["R2"].round(3),
            MAE=pm10_results["MAE"].round(2),
            RMSE=pm10_results["RMSE"].round(2)
        ),
        use_container_width=True
    )
    st.bar_chart(pm10_results.set_index("model")[["R2"]])

ap1, ap2 = st.columns(2)
with ap1:
    best_pm25 = pm25_results.iloc[0]["model"]
    st.pyplot(plot_actual_vs_predicted(pm25_preds[best_pm25], "PM2.5", best_pm25))
with ap2:
    best_pm10 = pm10_results.iloc[0]["model"]
    st.pyplot(plot_actual_vs_predicted(pm10_preds[best_pm10], "PM10", best_pm10))

# ========================= Section 3: Feature explanation =========================
st.subheader("3. Model Explanation: Random Forest Feature Importance")
fi1, fi2 = st.columns(2)
with fi1:
    pm25_imp = get_rf_feature_importance(df, "pm25")
    st.pyplot(plot_feature_importance(pm25_imp, "PM2.5"))
with fi2:
    pm10_imp = get_rf_feature_importance(df, "pm10")
    st.pyplot(plot_feature_importance(pm10_imp, "PM10"))

with st.expander("Top feature importance tables"):
    t1, t2 = st.columns(2)
    with t1:
        st.write("**PM2.5**")
        st.dataframe(pm25_imp.head(15), use_container_width=True)
    with t2:
        st.write("**PM10**")
        st.dataframe(pm10_imp.head(15), use_container_width=True)

# ========================= Section 4: Historical explorer =========================
if show_historical:
    st.subheader("4. Historical Training-Data Explorer")
    st.write(
        "This part is for explaining the 2023 data used to train the models. It is not the main prediction input."
    )

    df_hist = df.copy()
    df_hist["season_group"] = np.where(df_hist["fire_season"] == 1, "Fire Season", "Non-Fire Season")

    season_filter = st.selectbox("Historical season filter", ["All", "Fire Season", "Non-Fire Season"])
    if season_filter != "All":
        display_df = df_hist[df_hist["season_group"] == season_filter].copy()
    else:
        display_df = df_hist.copy()

    h1, h2 = st.columns(2)
    with h1:
        st.write("**Fire Activity Trend: Fire Count and Total FRP**")
        st.line_chart(display_df.set_index("date")[["fire_count", "total_frp"]])
    with h2:
        st.write("**Observed PM2.5 and PM10 Trend**")
        st.line_chart(display_df.set_index("date")[["pm25", "pm10"]])

    season_summary = (
        df_hist.groupby("season_group")
        .agg(
            mean_pm25=("pm25", "mean"),
            mean_pm10=("pm10", "mean"),
            mean_fire_count=("fire_count", "mean"),
            observed_unhealthy_pm25_days=("pm25", lambda x: (x > 100).mean() * 100)
        )
        .reset_index()
    )

    s1, s2 = st.columns(2)
    with s1:
        st.write("**Average PM2.5 and PM10 by Season**")
        st.bar_chart(season_summary.set_index("season_group")[["mean_pm25", "mean_pm10"]])
    with s2:
        st.write("**Observed Unhealthy PM2.5 Days by Season (%)**")
        st.bar_chart(season_summary.set_index("season_group")[["observed_unhealthy_pm25_days"]])
    st.dataframe(season_summary, use_container_width=True)

st.info(
    "Limitation note: This dashboard is an early-warning decision-support prototype, not a complete official warning system. "
    "The current model uses fire activity, lagged fire indicators, rolling fire indicators, and seasonal variables. "
    "Wind direction, rainfall, atmospheric transport, traffic emissions, industrial emissions, and health outcome data are not included."
)
