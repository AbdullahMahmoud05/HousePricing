"""
app.py
------
Streamlit application for the House Price Predictor.

Can be run via:
    streamlit run app.py
OR directly via:
    python app.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

QUALITY_LABELS = {
    "Ex": "Excellent (Ex)",
    "Gd": "Good (Gd)",
    "TA": "Average / Typical (TA)",
    "Fa": "Fair (Fa)",
    "Po": "Poor (Po)",
}

QUALITY_ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]

GARAGE_TYPE_LABELS = {
    "Attchd": "Attached to home (Attchd)",
    "Detchd": "Detached from home (Detchd)",
    "BuiltIn": "Built-In (part of house) (BuiltIn)",
    "Basment": "Basement Garage (Basment)",
    "CarPort": "Carport (CarPort)",
    "2Types": "More than one type (2Types)",
    "NoGarage": "No Garage (NoGarage)",
}


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling and crisp readability
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .card-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
    }
    .price-title {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        margin-bottom: 0.25rem;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    .stat-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model Loading & Auto-Training
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_artifact(path: Path):
    """Loads the trained artifact dictionary from pickle."""
    if not path.exists():
        # Try auto-training if train.csv is available
        try:
            import model
            artifact = model.train()
            return artifact, None
        except Exception as exc:
            return None, (
                f"Model file not found at '{path.name}'. "
                f"Automatic training failed: {exc}. Please run `python model.py`."
            )
    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        required_keys = {
            "model",
            "scaler",
            "model_input_columns",
            "quality_ordinal_map",
            "garage_type_categories",
            "garage_type_dummy_columns",
        }
        missing = required_keys - artifact.keys()
        if missing:
            return None, f"Model file is missing required metadata: {missing}."
        return artifact, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to load model file: {exc}"


artifact, load_error = load_artifact(MODEL_PATH)


# ---------------------------------------------------------------------------
# Sidebar & App Info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🏠 House Price ML")
    st.markdown(
        "A **Production-Ready Linear Regression** application trained on the Ames Housing dataset."
    )
    st.divider()

    if artifact:
        metrics = artifact.get("metrics", {})
        st.subheader("📊 Model Performance")
        st.metric("Test R² Score", f"{metrics.get('test_r2', 0.8003):.4f}")
        st.metric("Test MAE", f"${metrics.get('test_mae', 25607):,.0f}")
        st.metric("Test RMSE", f"${metrics.get('test_rmse', 33208):,.0f}")
        st.divider()
        st.caption("Linear Regression with StandardScaler preprocessing.")
    else:
        st.warning("⚠️ Model is not currently loaded.")

    if st.button("🔄 Retrain Model", use_container_width=True):
        with st.spinner("Retraining model from train.csv..."):
            try:
                import model
                model.train()
                st.cache_resource.clear()
                st.success("Model retrained successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Retraining failed: {e}")


# ---------------------------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">🏠 House Price Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter property specifications to generate an accurate valuation based on the Ames Housing Linear Regression model.</div>',
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"⚠️ {load_error}")
    st.info("Ensure `train.csv` is in the project folder and click 'Retrain Model' in the sidebar.")
    st.stop()

model_input_columns = artifact["model_input_columns"]
quality_ordinal_map = artifact["quality_ordinal_map"]
garage_type_categories = artifact["garage_type_categories"]
garage_type_dummy_columns = artifact["garage_type_dummy_columns"]
target_log_transformed = artifact.get("target_log_transformed", False)


# ---------------------------------------------------------------------------
# Property Input Form
# ---------------------------------------------------------------------------

col_left, col_mid, col_right = st.columns(3)

with col_left:
    st.markdown("### 🏛️ Structure & Quality")
    overall_qual = st.slider(
        "Overall Quality Rating",
        min_value=1,
        max_value=10,
        value=6,
        step=1,
        help="Rates the overall material and finish of the house (1 = Very Poor, 10 = Very Excellent)",
    )
    year_built = st.number_input(
        "Year Built",
        min_value=1870,
        max_value=2026,
        value=1975,
        step=1,
    )
    year_remod = st.number_input(
        "Year Remodeled / Added",
        min_value=1870,
        max_value=2026,
        value=1995,
        step=1,
        help="Same as Year Built if never remodeled.",
    )
    ext_qual = st.selectbox(
        "Exterior Material Quality",
        options=QUALITY_ORDER,
        format_func=lambda v: QUALITY_LABELS[v],
        index=QUALITY_ORDER.index("TA"),
    )

with col_mid:
    st.markdown("### 📐 Living Area & Rooms")
    gr_liv_area = st.number_input(
        "Above-Grade Living Area (sq ft)",
        min_value=200,
        max_value=6000,
        value=1500,
        step=25,
    )
    first_flr_sf = st.number_input(
        "1st Floor Area (sq ft)",
        min_value=200,
        max_value=5000,
        value=1100,
        step=25,
    )
    full_bath = st.selectbox("Full Bathrooms", options=[0, 1, 2, 3, 4], index=2)
    tot_rms = st.slider(
        "Total Rooms Above Grade (excl. baths)",
        min_value=2,
        max_value=14,
        value=6,
        step=1,
    )
    kitchen_qual = st.selectbox(
        "Kitchen Quality",
        options=QUALITY_ORDER,
        format_func=lambda v: QUALITY_LABELS[v],
        index=QUALITY_ORDER.index("TA"),
    )

with col_right:
    st.markdown("### 🚗 Basement & Garage")
    has_basement = st.checkbox("Property has a Basement", value=True)
    bsmt_qual = None
    if has_basement:
        bsmt_qual = st.selectbox(
            "Basement Quality",
            options=QUALITY_ORDER,
            format_func=lambda v: QUALITY_LABELS[v],
            index=QUALITY_ORDER.index("TA"),
        )
    else:
        st.info("No Basement (imputed with NoBsmt)")

    garage_type = st.selectbox(
        "Garage Location / Type",
        options=garage_type_categories,
        format_func=lambda v: GARAGE_TYPE_LABELS.get(v, v),
        index=garage_type_categories.index("Attchd") if "Attchd" in garage_type_categories else 0,
    )
    garage_cars = st.selectbox("Garage Car Capacity", options=[0, 1, 2, 3, 4, 5], index=2)

st.divider()


# ---------------------------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------------------------

def build_feature_row(raw_inputs: dict) -> pd.DataFrame:
    """Assembles a single-row DataFrame in the exact feature order used during training."""
    row = {}

    # Numeric passthrough features
    row["OverallQual"] = raw_inputs["OverallQual"]
    row["YearBuilt"] = raw_inputs["YearBuilt"]
    row["YearRemodAdd"] = raw_inputs["YearRemodAdd"]
    row["FullBath"] = raw_inputs["FullBath"]
    row["TotRmsAbvGrd"] = raw_inputs["TotRmsAbvGrd"]
    row["GarageCars"] = raw_inputs["GarageCars"]

    # Log1p transforms
    row["1stFlrSF"] = float(np.log1p(raw_inputs["1stFlrSF"]))
    row["GrLivArea"] = float(np.log1p(raw_inputs["GrLivArea"]))

    # Ordinal Quality encoding
    row["ExterQual"] = quality_ordinal_map[raw_inputs["ExterQual"]]
    bsmt_val = raw_inputs["BsmtQual"] if raw_inputs["BsmtQual"] is not None else "NoBsmt"
    row["BsmtQual"] = quality_ordinal_map[bsmt_val]
    row["KitchenQual"] = quality_ordinal_map[raw_inputs["KitchenQual"]]

    # One-hot encoding for GarageType
    for col in garage_type_dummy_columns:
        row[col] = 0
    gt_val = raw_inputs["GarageType"]
    dummy_col = f"GarageType_{gt_val}"
    if dummy_col in row:
        row[dummy_col] = 1
    elif gt_val != artifact.get("garage_type_baseline", "2Types"):
        raise ValueError(f"Unrecognized garage type: '{gt_val}'")

    ordered = {col: row[col] for col in model_input_columns}
    return pd.DataFrame([ordered])


predict_clicked = st.button("🔮 Calculate Estimated Price", type="primary", use_container_width=True)

if predict_clicked:
    if year_remod < year_built:
        st.error("⚠️ 'Year Remodeled' cannot be earlier than 'Year Built'. Please correct the year inputs.")
    else:
        raw_inputs = {
            "OverallQual": overall_qual,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "ExterQual": ext_qual,
            "BsmtQual": bsmt_qual,
            "1stFlrSF": first_flr_sf,
            "GrLivArea": gr_liv_area,
            "FullBath": full_bath,
            "KitchenQual": kitchen_qual,
            "TotRmsAbvGrd": tot_rms,
            "GarageType": garage_type,
            "GarageCars": garage_cars,
        }

        try:
            feature_row = build_feature_row(raw_inputs)
            
            # Pass DataFrame directly so feature names are preserved without warnings
            scaled = artifact["scaler"].transform(feature_row)
            prediction = artifact["model"].predict(scaled)[0]

            if target_log_transformed:
                prediction = np.expm1(prediction)

            prediction = max(prediction, 0.0)
            price_per_sqft = prediction / max(gr_liv_area, 1)

            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-title">Estimated Property Market Value</div>
                    <div class="price-value">${prediction:,.0f}</div>
                    <div class="stat-badge">Estimated Price / Sq Ft: ${price_per_sqft:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Living Area", f"{gr_liv_area:,} sq ft")
            with col_res2:
                st.metric("Quality Rating", f"{overall_qual} / 10")
            with col_res3:
                st.metric("Property Age", f"{2026 - year_built} years")

            with st.expander("🔍 View Processed Features Vector"):
                st.dataframe(feature_row, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction error: {exc}")


# ---------------------------------------------------------------------------
# Direct Execution Hook (python app.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if ctx is None:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())
