import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================================================
# Jakarta Fire-Air Quality Early Warning Dashboard
# Based on GA2 notebook modelling logic
# Targets: PM2.5 and PM10
# Models: Linear Regression, Ridge Regression, Decision Tree,
#         Random Forest, Gradient Boosting
# =========================================================

YEAR = 2023
RANDOM_STATE = 42

AIR_PATH = "ispu_dki_all.csv"
FIRE_PATH = "fire_archive_SV-C2_744546.csv"
PROCESSED_PATH = os.path.join("processed_data", "model_dataset_2023.csv")
MODEL_RESULTS_PATH = os.path.join("outputs", "model_results.csv")
FEATURE_IMPORTANCE_PATH = os.path.join("outputs", "feature_importance.csv")

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


def pm25_risk_category(value):
    if pd.isna(value):
        return "Not available"
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    return "Unhealthy"


def risk_message(category):
    messages = {
        "Good": "Predicted PM2.5 is Good. Routine monitoring is sufficient.",
        "Moderate": "Predicted PM2.5 is Moderate. Sensitive groups should pay attention.",
        "Unhealthy": "Predicted PM2.5 is Unhealthy. Increase monitoring and prepare public health advisory.",
        "Not available": "PM2.5 prediction is not available for this selected record."
    }
    return messages.get(category, "Risk category unavailable.")


@st.cache_data(show_spinner=False)
def build_dataset_from_raw():
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

    # Feature engineering
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["fire_season"] = df["month"].isin([8, 9, 10]).astype(int)

    for lag in [1, 2, 3, 7]:
        df[f"fire_count_lag{lag}"] = df["fire_count"].shift(lag).fillna(0)
        df[f"total_frp_lag{lag}"] = df["total_frp"].shift(lag).fillna(0)
        df[f"high_fire_count_lag{lag}"] = df["high_fire_count"].shift(lag).fillna(0)

    for window in [3, 7, 14]:
        df[f"fire_count_roll{window}"] = df["fire_count"].rolling(window, min_periods=1).mean()
        df[f"total_frp_roll{window}"] = df["total_frp"].rolling(window, min_periods=1).mean()
        df[f"high_fire_count_roll{window}"] = df["high_fire_count"].rolling(window, min_periods=1).mean()

    os.makedirs("processed_data", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return df


@st.cache_data(show_spinner=False)
def load_dataset():
    possible_paths = [PROCESSED_PATH, "model_dataset_2023.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
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

    for model_name, model in MODEL_DEFINITIONS.items():
        fitted = model.fit(X_train, y_train)
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
    model = MODEL_DEFINITIONS[model_name]
    model.fit(X, y)
    return model


def predict_for_all_dates(df, target_col, model_name):
    model = fit_full_model(df, target_col, model_name)
    pred = model.predict(df[FEATURE_COLS])
    return pred


def get_rf_feature_importance(df, target_col):
    model_name = "Random Forest"
    model = fit_full_model(df, target_col, model_name)
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
    ax.set_title(f"Top 10 Feature Importance for {target_label}")
    fig.tight_layout()
    return fig


# ========================= Streamlit UI =========================
st.set_page_config(
    page_title="Jakarta Fire-Air Quality Early Warning Dashboard",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Jakarta Fire-Air Quality Early Warning Dashboard")
st.caption(
    "This dashboard converts GA2 regression modelling results into a simple decision-support prototype "
    "for monitoring Indonesian fire activity and Jakarta particulate pollution risk."
)

with st.sidebar:
    st.header("Dashboard Controls")
    st.write("Jakarta Air Quality and Indonesian Fire Activity, 2023")

    df = load_dataset()

    if df is None:
        st.error(
            "Dataset not found. Please run ga2.ipynb first, or place either "
            "processed_data/model_dataset_2023.csv or the raw CSV files in this folder."
        )
        st.stop()

    target_choice = st.selectbox("Select pollutant target", ["pm25", "pm10"], format_func=lambda x: x.upper())
    model_choice = st.selectbox(
        "Select model for daily prediction",
        list(MODEL_DEFINITIONS.keys()),
        index=list(MODEL_DEFINITIONS.keys()).index("Random Forest")
    )

    available_dates = df["date"].dropna().sort_values().dt.date.tolist()
    selected_date = st.date_input(
        "Select a date",
        value=available_dates[-1],
        min_value=available_dates[0],
        max_value=available_dates[-1]
    )

    season_filter = st.selectbox("Season filter", ["All", "Fire Season", "Non-Fire Season"])
    show_table = st.checkbox("Show selected-date data table", value=False)

# Prepare data
working_df = df.copy()
working_df["pred_pm25"] = predict_for_all_dates(working_df, "pm25", model_choice)
working_df["pred_pm10"] = predict_for_all_dates(working_df, "pm10", model_choice)
working_df["pred_pm25_risk"] = working_df["pred_pm25"].apply(pm25_risk_category)
working_df["season_group"] = np.where(working_df["fire_season"] == 1, "Fire Season", "Non-Fire Season")

if season_filter != "All":
    display_df = working_df[working_df["season_group"] == season_filter].copy()
else:
    display_df = working_df.copy()

selected_ts = pd.to_datetime(selected_date)
selected_rows = working_df[working_df["date"].dt.date == selected_date]

if selected_rows.empty:
    st.warning("No record is available for the selected date.")
    st.stop()

row = selected_rows.iloc[0]
risk = row["pred_pm25_risk"]

# Section 1: Daily overview
st.subheader("1. Daily Overview")
metric_cols = st.columns(6)
metric_cols[0].metric("Selected Date", selected_date.strftime("%Y-%m-%d"))
metric_cols[1].metric("Fire Count", f"{row['fire_count']:.0f}")
metric_cols[2].metric("Total FRP", f"{row['total_frp']:.1f}")
metric_cols[3].metric("Predicted PM2.5", f"{row['pred_pm25']:.2f}")
metric_cols[4].metric("Predicted PM10", f"{row['pred_pm10']:.2f}")
metric_cols[5].metric("PM2.5 Risk", risk)

if risk == "Unhealthy":
    st.error(risk_message(risk))
elif risk == "Moderate":
    st.warning(risk_message(risk))
else:
    st.success(risk_message(risk))

if show_table:
    selected_cols = [
        "date", "fire_count", "total_frp", "avg_frp", "max_frp",
        "pm25", "pm10", "pred_pm25", "pred_pm10", "pred_pm25_risk"
    ]
    st.dataframe(selected_rows[selected_cols], use_container_width=True)

# Section 2: Trends
st.subheader("2. Daily Trend Monitoring")
left, right = st.columns(2)
with left:
    st.write("**Fire Activity Trend: Fire Count and Total FRP**")
    trend_fire = display_df.set_index("date")[["fire_count", "total_frp"]]
    st.line_chart(trend_fire)

with right:
    st.write("**Observed and Predicted PM2.5 / PM10**")
    trend_air_cols = ["pm25", "pm10", "pred_pm25", "pred_pm10"]
    trend_air = display_df.set_index("date")[trend_air_cols]
    st.line_chart(trend_air)

# Section 3: Seasonal comparison
st.subheader("3. Fire Season vs Non-Fire Season Comparison")
season_summary = (
    working_df.groupby("season_group")
    .agg(
        mean_pm25=("pm25", "mean"),
        mean_pm10=("pm10", "mean"),
        mean_fire_count=("fire_count", "mean"),
        unhealthy_pm25_days=("pred_pm25_risk", lambda x: (x == "Unhealthy").mean() * 100)
    )
    .reset_index()
)

c1, c2 = st.columns(2)
with c1:
    st.write("**Average PM2.5 and PM10 by Season**")
    st.bar_chart(season_summary.set_index("season_group")[["mean_pm25", "mean_pm10"]])
with c2:
    st.write("**Percentage of Predicted Unhealthy PM2.5 Days by Season**")
    st.bar_chart(season_summary.set_index("season_group")[["unhealthy_pm25_days"]])
st.dataframe(season_summary, use_container_width=True)

# Section 4: Model performance
st.subheader("4. Model Performance and Prediction Output")
results_df, pred_dict = evaluate_models(working_df, target_choice)
target_label = target_choice.upper()

m1, m2 = st.columns([1, 1])
with m1:
    st.write(f"**Model Evaluation for {target_label}**")
    st.dataframe(
        results_df.assign(
            R2=results_df["R2"].round(3),
            MAE=results_df["MAE"].round(2),
            RMSE=results_df["RMSE"].round(2)
        ),
        use_container_width=True
    )

    st.write(f"**Model R² Comparison for {target_label}**")
    st.bar_chart(results_df.set_index("model")[["R2"]])

with m2:
    best_name = results_df.iloc[0]["model"]
    st.write(f"**Actual vs Predicted: {target_label} ({best_name})**")
    st.pyplot(plot_actual_vs_predicted(pred_dict[best_name], target_label, best_name))

# Section 5: Feature explanation
st.subheader("5. Model Explanation: Feature Importance")
st.write(
    "Feature importance is shown for Random Forest because it is the main interpretable ensemble model "
    "used in the GA2 modelling design."
)
importance_df = get_rf_feature_importance(working_df, target_choice)
fi_left, fi_right = st.columns([1, 1])
with fi_left:
    st.pyplot(plot_feature_importance(importance_df, target_label))
with fi_right:
    st.dataframe(importance_df.head(15), use_container_width=True)

st.info(
    "Limitation note: This dashboard is an early-warning aid, not a complete official warning system. "
    "The current model uses fire activity, lagged fire indicators, rolling fire indicators, and seasonal variables. "
    "Wind direction, rainfall, atmospheric transport, traffic emissions, industrial emissions, and health outcome data are not included."
)
