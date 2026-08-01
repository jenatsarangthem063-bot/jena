"""
logic.py
Core data engineering, modeling, and simulation logic for the
Nassau Candy Factory Reallocation & Shipping Optimization system.

Kept free of any Streamlit imports so it can be unit-tested and reused.
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Static reference data (from the Nassau Candy project brief)
# ---------------------------------------------------------------------------

FACTORY_COORDS = {
    "Lot's O' Nuts":       (32.881893, -111.768036),
    "Wicked Choccy's":     (32.076176,  -81.088371),
    "Sugar Shack":         (48.119140,  -96.181150),
    "Secret Factory":      (41.446333,  -90.565487),
    "The Other Factory":   (35.117500,  -89.971107),
}

FACTORIES = list(FACTORY_COORDS.keys())

# Approximate geographic-center coordinates for US states / Canadian provinces
# present in the dataset. Used only as a relative distance proxy for the
# shipping-lead-time & logistics-cost model (not for turn-by-turn routing).
STATE_COORDS = {
    "Alabama": (32.806671, -86.791130), "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123), "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104), "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141), "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783), "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828), "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278), "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486), "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805), "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101), "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095), "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696), "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353), "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374), "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011), "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051), "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012), "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917), "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755), "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007), "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345), "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434), "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968), "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453), "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    # Canadian provinces
    "Ontario": (51.253775, -85.323214), "Quebec": (52.939916, -73.549136),
    "Nova Scotia": (44.681990, -63.744311), "Alberta": (53.933271, -116.576504),
    "British Columbia": (53.726669, -127.647621), "Manitoba": (53.760860, -98.813873),
    "Saskatchewan": (52.939916, -106.450864), "New Brunswick": (46.565314, -66.461914),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "Prince Edward Island": (46.510712, -63.416817),
}

# Assumption used throughout the simulator: incremental logistics cost per
# unit per mile. Documented here so it is transparent / tunable in one place.
SHIPPING_RATE_PER_UNIT_MILE = 0.0025


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_and_engineer(csv_path):
    """Load the cleaned Nassau dataset and engineer distance / date features."""
    df = pd.read_csv(csv_path)

    # Drop the pre-existing 'Recommended Factory' column if present: in the
    # supplied cleaned file it is a single constant value for every row and
    # carries no real signal, so this app derives its own recommendations.
    if "Recommended Factory" in df.columns:
        df = df.drop(columns=["Recommended Factory"])

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    # Distance from the CURRENTLY assigned factory to the customer's state
    def dist_current(row):
        s_lat, s_lon = STATE_COORDS.get(row["State/Province"], (np.nan, np.nan))
        f_lat, f_lon = FACTORY_COORDS.get(row["Factory"], (np.nan, np.nan))
        if np.isnan(s_lat):
            return np.nan
        return haversine_miles(f_lat, f_lon, s_lat, s_lon)

    df["Distance"] = df.apply(dist_current, axis=1)

    # Pre-compute distance from EVERY factory to each row's customer state so
    # scenario simulation doesn't need to re-derive coordinates repeatedly.
    for fac, (f_lat, f_lon) in FACTORY_COORDS.items():
        col = f"Dist::{fac}"

        def _d(row, f_lat=f_lat, f_lon=f_lon):
            s_lat, s_lon = STATE_COORDS.get(row["State/Province"], (np.nan, np.nan))
            if np.isnan(s_lat):
                return np.nan
            return haversine_miles(f_lat, f_lon, s_lat, s_lon)

        df[col] = df.apply(_d, axis=1)

    df = df.dropna(subset=["Distance"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Predictive modeling: Lead Time
# ---------------------------------------------------------------------------

FEATURE_COLS = ["Division", "Factory", "Region", "Ship Mode", "Distance", "Units"]
TARGET_COL = "Lead Time"


def build_feature_matrix(df, encoders=None):
    """Label-encode categoricals and assemble the numeric feature matrix.
    If `encoders` is None, fit new encoders (training path); otherwise reuse
    the supplied encoders (inference path) and clip unseen categories to a
    known label to avoid crashes.
    """
    cat_cols = ["Division", "Factory", "Region", "Ship Mode"]
    df = df.copy()
    fitted = {}
    for c in cat_cols:
        if encoders is None:
            le = LabelEncoder()
            df[c + "_enc"] = le.fit_transform(df[c].astype(str))
            fitted[c] = le
        else:
            le = encoders[c]
            df[c + "_enc"] = df[c].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
            fitted[c] = le

    X = df[["Division_enc", "Factory_enc", "Region_enc", "Ship Mode_enc", "Distance", "Units"]].copy()
    X.columns = ["Division", "Factory", "Region", "Ship Mode", "Distance", "Units"]
    return X, fitted


def train_models(df, random_state=42):
    """Train Linear Regression, Random Forest, and Gradient Boosting models
    to predict shipping Lead Time, and return metrics + the best model.
    """
    X, encoders = build_feature_matrix(df, encoders=None)
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=random_state),
    }

    metrics = []
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        metrics.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        fitted_models[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    # Feature importance (tree models only)
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Hold out predictions for an actual-vs-predicted plot
    test_results = pd.DataFrame({"Actual": y_test, "Predicted": fitted_models[best_name].predict(X_test)})

    return {
        "metrics": metrics_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "all_models": fitted_models,
        "encoders": encoders,
        "importances": importances,
        "test_results": test_results,
    }


def predict_lead_time(model, encoders, division, factory, region, ship_mode, distance, units):
    """Predict lead time for a single hypothetical configuration."""
    row = pd.DataFrame([{
        "Division": division, "Factory": factory, "Region": region,
        "Ship Mode": ship_mode, "Distance": distance, "Units": units,
    }])
    X, _ = build_feature_matrix(row, encoders=encoders)
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# Route & Product clustering
# ---------------------------------------------------------------------------

def cluster_routes(df, n_clusters=4, random_state=42):
    """Cluster (Region, Product Name) combinations by average lead time and
    average profit margin to surface consistently slow / congested routes.
    """
    grp = df.groupby(["Region", "Product Name"]).agg(
        Avg_Lead_Time=("Lead Time", "mean"),
        Avg_Profit_Margin=("Profit Margin", "mean"),
        Total_Units=("Units", "sum"),
        Orders=("Order ID", "count"),
    ).reset_index()

    feats = grp[["Avg_Lead_Time", "Avg_Profit_Margin"]].copy()
    feats_norm = (feats - feats.mean()) / feats.std()

    k = min(n_clusters, max(2, grp.shape[0] // 2)) if grp.shape[0] >= 4 else 2
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    grp["Cluster"] = km.fit_predict(feats_norm)

    # Label clusters by their centroid characteristics
    centroids = grp.groupby("Cluster")[["Avg_Lead_Time", "Avg_Profit_Margin"]].mean()
    lead_med = centroids["Avg_Lead_Time"].median()
    margin_med = centroids["Avg_Profit_Margin"].median()

    def label_for(row):
        slow = row["Avg_Lead_Time"] >= lead_med
        low_margin = row["Avg_Profit_Margin"] < margin_med
        if slow and low_margin:
            return "Congested / High-Risk"
        if slow and not low_margin:
            return "Slow but Profitable"
        if not slow and low_margin:
            return "Fast but Thin-Margin"
        return "Efficient"

    centroids["Label"] = centroids.apply(label_for, axis=1)
    grp["Cluster Label"] = grp["Cluster"].map(centroids["Label"])
    return grp


# ---------------------------------------------------------------------------
# Scenario simulation engine
# ---------------------------------------------------------------------------

def simulate_reassignment(subset_df, model, encoders, priority_weight=0.5,
                           shipping_rate=SHIPPING_RATE_PER_UNIT_MILE):
    """Given a filtered slice of the dataset (e.g. one product, optionally
    further filtered by region / ship mode), evaluate every candidate
    factory and return a ranked comparison table plus the current baseline.

    priority_weight: 0 = optimize purely for profit, 1 = optimize purely for
    speed (lead-time reduction). 0.5 = balanced.
    """
    if subset_df.empty:
        return None

    division = subset_df["Division"].mode()[0]
    current_factory = subset_df["Factory"].mode()[0]
    dominant_region = subset_df["Region"].mode()[0]
    dominant_ship_mode = subset_df["Ship Mode"].mode()[0]

    total_units = float(subset_df["Units"].sum())
    total_sales = float(subset_df["Sales"].sum())
    total_cost = float(subset_df["Cost"].sum())
    total_profit = float(subset_df["Gross Profit"].sum())
    avg_units = float(subset_df["Units"].mean())
    order_count = int(subset_df.shape[0])

    dist_cols = {fac: f"Dist::{fac}" for fac in FACTORIES}
    weighted_dist = {}
    for fac, col in dist_cols.items():
        w = subset_df["Units"]
        weighted_dist[fac] = float((subset_df[col] * w).sum() / w.sum())

    current_distance = weighted_dist[current_factory]
    manufacturing_cost_only = total_cost - (current_distance * shipping_rate * total_units)

    rows = []
    for fac in FACTORIES:
        distance = weighted_dist[fac]
        pred_lead_time = predict_lead_time(
            model, encoders, division, fac, dominant_region, dominant_ship_mode,
            distance, avg_units,
        )
        shipping_cost = distance * shipping_rate * total_units
        simulated_cost = manufacturing_cost_only + shipping_cost
        simulated_profit = total_sales - simulated_cost
        profit_delta = simulated_profit - total_profit
        rows.append({
            "Factory": fac,
            "Is Current": fac == current_factory,
            "Distance (mi)": round(distance, 1),
            "Predicted Lead Time (days)": round(pred_lead_time, 1),
            "Simulated Profit ($)": round(simulated_profit, 2),
            "Profit Impact ($)": round(profit_delta, 2),
            "Profit Impact (%)": round((profit_delta / total_profit * 100) if total_profit else 0, 2),
        })

    result = pd.DataFrame(rows)
    current_row = result[result["Is Current"]].iloc[0]

    result["Lead Time Reduction (%)"] = (
        (current_row["Predicted Lead Time (days)"] - result["Predicted Lead Time (days)"])
        / current_row["Predicted Lead Time (days)"] * 100
    ).round(2)

    # Normalize the two objectives (0-1) for a blended ranking score
    lt = result["Lead Time Reduction (%)"]
    pi = result["Profit Impact (%)"]
    lt_norm = (lt - lt.min()) / (lt.max() - lt.min()) if lt.max() != lt.min() else lt * 0
    pi_norm = (pi - pi.min()) / (pi.max() - pi.min()) if pi.max() != pi.min() else pi * 0
    result["Score"] = (priority_weight * lt_norm + (1 - priority_weight) * pi_norm).round(4)

    # Simple confidence score from sample size supporting the estimate
    confidence = min(100, 40 + math.log1p(order_count) * 12)
    result["Confidence Score"] = round(confidence, 1)
    result["Risk Flag"] = np.where(result["Profit Impact ($)"] < -0.05 * abs(total_profit if total_profit else 1),
                                    "High Risk", "Normal")

    result = result.sort_values("Score", ascending=False).reset_index(drop=True)

    meta = {
        "division": division, "current_factory": current_factory,
        "dominant_region": dominant_region, "dominant_ship_mode": dominant_ship_mode,
        "total_units": total_units, "total_sales": total_sales,
        "total_profit": total_profit, "order_count": order_count,
    }
    return result, meta


def batch_recommendations(df, model, encoders, priority_weight=0.5):
    """Run the scenario simulator for every product (across its full
    customer base) and return one recommendation row per product.
    """
    out = []
    for product in df["Product Name"].unique():
        subset = df[df["Product Name"] == product]
        sim, meta = simulate_reassignment(subset, model, encoders, priority_weight)
        top = sim.iloc[0]
        current = sim[sim["Is Current"]].iloc[0]
        recommended_factory = top["Factory"]
        changed = recommended_factory != meta["current_factory"]
        out.append({
            "Product Name": product,
            "Division": meta["division"],
            "Current Factory": meta["current_factory"],
            "Recommended Factory": recommended_factory,
            "Changed?": "Reassign" if changed else "Keep Current",
            "Lead Time Reduction (%)": top["Lead Time Reduction (%)"],
            "Profit Impact (%)": top["Profit Impact (%)"],
            "Profit Impact ($)": top["Profit Impact ($)"],
            "Confidence Score": top["Confidence Score"],
            "Risk Flag": top["Risk Flag"],
            "Order Count": meta["order_count"],
        })
    return pd.DataFrame(out).sort_values("Profit Impact ($)", ascending=False).reset_index(drop=True)
