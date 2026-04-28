"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       Breast Cancer Diagnosis — Production Streamlit App                    ║
║       Model: Support Vector Classifier (SVC)                                ║
║       Dataset: Breast Cancer Wisconsin                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings
from sklearn.svm              import SVC
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, roc_auc_score, roc_curve,
                                       ConfusionMatrixDisplay)
from sklearn.inspection       import permutation_importance
warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="🔬 Cancer Diagnosis AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CUSTOM CSS                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    border-right: 1px solid #e94560;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
    color: #e0e0e0 !important;
}

/* ── Header ────────────────────────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e, #0f3460, #16213e);
    padding: 2.5rem 2rem;
    border-radius: 15px;
    border: 1px solid #e94560;
    text-align: center;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: #ffffff; font-size: 2.2rem; margin: 0; }
.main-header p  { color: #a0b4c8; margin: 0.4rem 0 0; font-size: 1rem; }

/* ── Cards ─────────────────────────────────────────────────────────────────── */
.metric-card {
    background: rgba(15,52,96,0.45);
    border: 1px solid #e94560;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 0.8rem;
}
.metric-card .val  { font-size: 2rem; font-weight: 700; color: #e94560; }
.metric-card .lbl  { font-size: 0.85rem; color: #a0b4c8; margin-top: 0.2rem; }

.prediction-box {
    background: rgba(15,52,96,0.6);
    border-radius: 15px;
    padding: 1.8rem;
    border: 2px solid;
    text-align: center;
    margin: 1rem 0;
}
.prediction-malignant { border-color: #e94560; }
.prediction-benign    { border-color: #00b894; }
.pred-label  { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.4rem; }
.pred-prob   { font-size: 1rem; color: #a0b4c8; }

.info-box {
    background: rgba(15,52,96,0.35);
    border-left: 4px solid #e94560;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}

.section-header {
    color: #e94560;
    font-size: 1.3rem;
    font-weight: 700;
    border-bottom: 2px solid #e94560;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* ── Slider override ────────────────────────────────────────────────────────── */
.stSlider > div > div > div { background: #e94560 !important; }

/* ── Button ─────────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(233,69,96,0.4); }

/* ── Tabs ───────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: rgba(15,52,96,0.3); border-radius: 10px; }
.stTabs [data-baseweb="tab"]      { color: #a0b4c8; }
.stTabs [aria-selected="true"]    { color: #e94560 !important; border-bottom: 2px solid #e94560; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
MODEL_PATH = "models/svc_breast_cancer.pkl"
META_PATH  = "models/svc_metadata.pkl"
DATA_PATH  = "breast-cancer.csv"

FEATURE_GROUPS = {
    "Mean Features": [
        "radius_mean","texture_mean","perimeter_mean","area_mean",
        "smoothness_mean","compactness_mean","concavity_mean",
        "concave points_mean","symmetry_mean","fractal_dimension_mean"
    ],
    "SE Features": [
        "radius_se","texture_se","perimeter_se","area_se",
        "smoothness_se","compactness_se","concavity_se",
        "concave points_se","symmetry_se","fractal_dimension_se"
    ],
    "Worst Features": [
        "radius_worst","texture_worst","perimeter_worst","area_worst",
        "smoothness_worst","compactness_worst","concavity_worst",
        "concave points_worst","symmetry_worst","fractal_dimension_worst"
    ],
}

FEATURE_DEFAULTS = {
    "radius_mean":14.13,"texture_mean":19.27,"perimeter_mean":91.97,"area_mean":654.9,
    "smoothness_mean":0.096,"compactness_mean":0.104,"concavity_mean":0.088,
    "concave points_mean":0.048,"symmetry_mean":0.181,"fractal_dimension_mean":0.062,
    "radius_se":0.405,"texture_se":1.217,"perimeter_se":2.866,"area_se":40.34,
    "smoothness_se":0.007,"compactness_se":0.025,"concavity_se":0.032,
    "concave points_se":0.012,"symmetry_se":0.020,"fractal_dimension_se":0.003,
    "radius_worst":16.27,"texture_worst":25.68,"perimeter_worst":107.26,"area_worst":880.6,
    "smoothness_worst":0.132,"compactness_worst":0.254,"concavity_worst":0.272,
    "concave points_worst":0.115,"symmetry_worst":0.290,"fractal_dimension_worst":0.084,
}

FEATURE_RANGES = {
    "radius_mean":(6.981,28.11),"texture_mean":(9.71,39.28),
    "perimeter_mean":(43.79,188.5),"area_mean":(143.5,2501.0),
    "smoothness_mean":(0.053,0.163),"compactness_mean":(0.019,0.345),
    "concavity_mean":(0.0,0.427),"concave points_mean":(0.0,0.201),
    "symmetry_mean":(0.106,0.304),"fractal_dimension_mean":(0.05,0.097),
    "radius_se":(0.112,2.873),"texture_se":(0.36,4.885),
    "perimeter_se":(0.757,21.98),"area_se":(6.802,542.2),
    "smoothness_se":(0.002,0.031),"compactness_se":(0.002,0.135),
    "concavity_se":(0.0,0.396),"concave points_se":(0.0,0.053),
    "symmetry_se":(0.008,0.079),"fractal_dimension_se":(0.001,0.03),
    "radius_worst":(7.93,36.04),"texture_worst":(12.02,49.54),
    "perimeter_worst":(50.41,251.2),"area_worst":(185.2,4254.0),
    "smoothness_worst":(0.071,0.223),"compactness_worst":(0.027,1.058),
    "concavity_worst":(0.0,1.252),"concave points_worst":(0.0,0.291),
    "symmetry_worst":(0.156,0.664),"fractal_dimension_worst":(0.055,0.208),
}

PALETTE_DARK = {"background": "#0d0d1a", "accent": "#e94560", "secondary": "#0f3460"}

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Load saved model if it exists; otherwise train from scratch."""
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        meta  = joblib.load(META_PATH)
        return model, meta, False   # False = not freshly trained

    # ── Train from scratch ────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        return None, None, None     # signal: data not found

    df = pd.read_csv(DATA_PATH)
    df.drop(columns=["id"], inplace=True, errors="ignore")

    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(probability=True, random_state=42))
    ])
    param_grid = {
        "svc__C":      [0.1, 1, 10],
        "svc__kernel": ["rbf", "linear"],
        "svc__gamma":  ["scale", "auto"]
    }
    gs = GridSearchCV(pipeline, param_grid,
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                      scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred      = best.predict(X_test)
    y_pred_prob = best.predict_proba(X_test)[:, 1]

    os.makedirs("models", exist_ok=True)
    meta = {
        "feature_names": X.columns.tolist(),
        "label_encoder": le,
        "best_params":   gs.best_params_,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc":  float(roc_auc_score(y_test, y_pred_prob)),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
    }
    joblib.dump(best, MODEL_PATH)
    joblib.dump(meta, META_PATH)
    return best, meta, True   # True = freshly trained


@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=["id"], inplace=True, errors="ignore")
    return df


def make_prediction(model, input_df):
    pred       = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    label      = "Malignant 🔴" if pred == 1 else "Benign 🟢"
    confidence = proba[pred]
    return pred, label, proba, confidence


def plot_confusion_matrix(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu_r",
                xticklabels=["Benign","Malignant"],
                yticklabels=["Benign","Malignant"],
                linewidths=1, linecolor="#0f3460",
                annot_kws={"size": 14, "color": "white"}, ax=ax)
    ax.set_xlabel("Predicted",  color="white", fontsize=11)
    ax.set_ylabel("Actual",     color="white", fontsize=11)
    ax.set_title("Confusion Matrix", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    plt.tight_layout()
    return fig


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val     = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.plot(fpr, tpr, color="#e94560", lw=2, label=f"AUC = {auc_val:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e94560")
    ax.plot([0,1],[0,1], "--", color="#555", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", color="white", fontsize=10)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=10)
    ax.set_title("ROC Curve", color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#0f3460", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#0f3460")
    plt.tight_layout()
    return fig


def plot_probability_gauge(proba):
    categories = ["Benign", "Malignant"]
    colors     = ["#00b894", "#e94560"]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    bars = ax.barh(categories, proba, color=colors, edgecolor="none",
                   height=0.5, alpha=0.85)
    for bar, val in zip(bars, proba):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", color="white", fontsize=13,
                fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_title("Prediction Probability", color="white",
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.axvline(0.5, color="#555", linestyle="--", lw=1.5)
    plt.tight_layout()
    return fig


def plot_feature_radar(input_vals, feature_names):
    """Radar chart comparing user input vs dataset mean."""
    df = load_dataset()
    if df is None:
        return None
    mean_cols = [c for c in feature_names if c.endswith("_mean")]
    if not mean_cols:
        return None
    df_enc = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_enc["diagnosis"] = le.fit_transform(df_enc["diagnosis"])

    benign_mean    = df_enc[df_enc["diagnosis"] == 0][mean_cols].mean().values
    malignant_mean = df_enc[df_enc["diagnosis"] == 1][mean_cols].mean().values
    user_vals      = np.array([input_vals.get(f, 0) for f in mean_cols])

    # Normalise 0-1 for radar
    max_v = np.maximum(benign_mean, malignant_mean)
    max_v[max_v == 0] = 1
    b_norm = benign_mean / max_v
    m_norm = malignant_mean / max_v
    u_norm = user_vals / max_v

    angles = np.linspace(0, 2*np.pi, len(mean_cols), endpoint=False).tolist()
    angles += angles[:1]

    def close(arr):
        return arr.tolist() + [arr[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.plot(angles, close(b_norm), "#00b894", lw=1.5, label="Benign Avg")
    ax.fill(angles, close(b_norm), "#00b894", alpha=0.1)
    ax.plot(angles, close(m_norm), "#e94560", lw=1.5, label="Malignant Avg")
    ax.fill(angles, close(m_norm), "#e94560", alpha=0.1)
    ax.plot(angles, close(u_norm), "#f9ca24", lw=2.5, label="Your Input", linestyle="--")
    short_labels = [f.replace("_mean","").replace("_"," ") for f in mean_cols]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, color="white", size=7)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#0f3460")
    ax.grid(color="#0f3460", linestyle="--", linewidth=0.5)
    ax.set_title("Feature Radar (Mean)", color="white",
                 fontsize=11, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#0f3460", labelcolor="white", fontsize=8)
    plt.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0;'>
            <h1 style='color:#e94560;font-size:1.8rem;margin:0;'>🔬 Cancer AI</h1>
            <p style='color:#a0b4c8;font-size:0.85rem;margin:0.3rem 0 0;'>
            Breast Cancer Diagnosis Tool</p>
        </div>
        <hr style='border:1px solid #e94560;margin:1rem 0;'>
        """, unsafe_allow_html=True)

        page = st.radio(
            "📌 Navigation",
            ["🏠 Home", "🔍 Predict (Manual)", "📁 Batch Predict", "📊 Model Insights"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border:1px solid #0f3460;'>", unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box'>
        <b style='color:#e94560;'>ℹ️ About</b><br>
        <span style='font-size:0.82rem;color:#c0cce0;'>
        Uses a Support Vector Classifier (SVC) trained on the Breast Cancer 
        Wisconsin dataset. Features are scaled via StandardScaler inside a 
        sklearn Pipeline optimised with GridSearchCV.
        </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box' style='margin-top:0.5rem;'>
        <b style='color:#e94560;'>⚠️ Disclaimer</b><br>
        <span style='font-size:0.82rem;color:#c0cce0;'>
        This tool is for <b>educational purposes only</b>. Do not use it as 
        a substitute for professional medical advice.
        </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#555;font-size:0.75rem;'>v1.0 · Built with Streamlit</p>", unsafe_allow_html=True)

    return page


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: HOME                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_home(model, meta):
    st.markdown("""
    <div class='main-header'>
        <h1>🔬 Breast Cancer Diagnosis AI</h1>
        <p>Support Vector Classifier · Wisconsin Dataset · Production-Ready</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{meta['test_accuracy']*100:.1f}%</div>
            <div class='lbl'>Test Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{meta['test_roc_auc']:.3f}</div>
            <div class='lbl'>ROC-AUC Score</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>569</div>
            <div class='lbl'>Training Samples</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>30</div>
            <div class='lbl'>Input Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset overview ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='section-header'>📖 Project Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        <b>Goal:</b> Classify breast tumours as <b style='color:#e94560;'>Malignant</b> or 
        <b style='color:#00b894;'>Benign</b> based on 30 nuclear morphology features computed 
        from digitised Fine Needle Aspirate (FNA) images.
        </div>

        <div class='info-box' style='margin-top:0.5rem;'>
        <b>Pipeline:</b><br>
        &nbsp;&nbsp;1. StandardScaler → removes scale bias<br>
        &nbsp;&nbsp;2. SVC (kernel=RBF) → maximum margin classification<br>
        &nbsp;&nbsp;3. GridSearchCV (5-fold CV) → optimal C, gamma, kernel
        </div>

        <div class='info-box' style='margin-top:0.5rem;'>
        <b>Best Hyperparameters:</b><br>
        """, unsafe_allow_html=True)
        for k, v in meta.get("best_params", {}).items():
            st.markdown(f"&nbsp;&nbsp;&nbsp;`{k}` = **{v}**")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        df = load_dataset()
        if df is not None:
            st.markdown("<div class='section-header'>📊 Dataset Snapshot</div>", unsafe_allow_html=True)

            dist = df["diagnosis"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            colors = ["#00b894", "#e94560"]
            bars = ax.bar(["Benign", "Malignant"], dist.values,
                          color=colors, edgecolor="none", alpha=0.85, width=0.5)
            for bar, val in zip(bars, dist.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 4,
                        f"{val}\n({val/len(df)*100:.1f}%)",
                        ha="center", color="white", fontsize=11, fontweight="bold")
            ax.set_title("Class Distribution", color="white",
                         fontsize=11, fontweight="bold")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#0f3460")
            ax.set_ylabel("Count", color="white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── Confusion matrix + ROC ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Model Performance on Test Set</div>",
                unsafe_allow_html=True)

    if "X_test" in meta:
        y_test = meta["y_test"]
        y_pred = meta["y_pred"]
        y_prob = meta["y_pred_prob"]
    else:
        # Re-generate from dataset
        df = load_dataset()
        if df is not None:
            le = LabelEncoder()
            df["diagnosis"] = le.fit_transform(df["diagnosis"])
            X = df.drop(columns=["diagnosis"])
            y = df["diagnosis"]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.20,
                                                     random_state=42, stratify=y)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_test = y_pred = y_prob = None

    if y_test is not None:
        cm_col, roc_col = st.columns(2)
        with cm_col:
            st.pyplot(plot_confusion_matrix(y_test, y_pred), use_container_width=True)
            plt.close()
        with roc_col:
            st.pyplot(plot_roc(y_test, y_prob), use_container_width=True)
            plt.close()

        # Classification report
        st.markdown("<div class='section-header'>📋 Classification Report</div>",
                    unsafe_allow_html=True)
        report = classification_report(y_test, y_pred,
                                       target_names=["Benign","Malignant"],
                                       output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(
            report_df.style
                .background_gradient(cmap="RdBu_r", subset=["precision","recall","f1-score"])
                .format("{:.3f}", subset=["precision","recall","f1-score"]),
            use_container_width=True
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: MANUAL PREDICT                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_manual_predict(model, meta):
    st.markdown("""
    <div class='main-header'>
        <h1>🔍 Manual Prediction</h1>
        <p>Adjust feature sliders and get an instant diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    feature_names = meta["feature_names"]
    input_vals    = {}

    st.markdown("<div class='info-box'>🎚️ Use the sliders below to enter tumour measurements. Hover over feature names for descriptions.</div>",
                unsafe_allow_html=True)

    # ── Feature input tabs ────────────────────────────────────────────────────
    tabs = st.tabs(list(FEATURE_GROUPS.keys()))
    for tab, (group_name, feats) in zip(tabs, FEATURE_GROUPS.items()):
        with tab:
            cols = st.columns(2)
            for i, feat in enumerate(feats):
                if feat not in feature_names:
                    continue
                lo, hi = FEATURE_RANGES.get(feat, (0.0, 1.0))
                default = FEATURE_DEFAULTS.get(feat, (lo + hi) / 2)
                with cols[i % 2]:
                    val = st.slider(
                        label=feat.replace("_", " ").title(),
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(default),
                        step=float((hi - lo) / 200),
                        key=f"slider_{feat}"
                    )
                    input_vals[feat] = val

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _, _ = st.columns([1, 1, 1])
    with col_btn:
        predict_clicked = st.button("🔬 Run Diagnosis")

    if predict_clicked:
        input_df = pd.DataFrame([{f: input_vals[f] for f in feature_names}])

        with st.spinner("Analysing tumour features …"):
            time.sleep(0.4)
            pred, label, proba, confidence = make_prediction(model, input_df)

        cls = "prediction-malignant" if pred == 1 else "prediction-benign"
        icon = "🔴" if pred == 1 else "🟢"
        col_box, col_radar = st.columns([1, 1])

        with col_box:
            st.markdown(f"""
            <div class='prediction-box {cls}'>
                <div class='pred-label'>{icon} {("Malignant" if pred==1 else "Benign")}</div>
                <div class='pred-prob'>Confidence: <b>{confidence*100:.1f}%</b></div>
                <br>
                <div style='font-size:0.9rem;color:#a0b4c8;'>
                {"⚠️ Immediate clinical evaluation recommended." if pred==1
                 else "✅ No malignant indicators detected."}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.pyplot(plot_probability_gauge(proba), use_container_width=True)
            plt.close()

        with col_radar:
            radar_fig = plot_feature_radar(input_vals, feature_names)
            if radar_fig:
                st.pyplot(radar_fig, use_container_width=True)
                plt.close()

        # Top influencing features
        st.markdown("<div class='section-header'>🧠 Input vs Dataset Mean</div>",
                    unsafe_allow_html=True)
        df_ref = load_dataset()
        if df_ref is not None:
            le_tmp = LabelEncoder()
            df_ref["diagnosis"] = le_tmp.fit_transform(df_ref["diagnosis"])
            mean_vals = df_ref[feature_names].mean()
            comp_df   = pd.DataFrame({
                "Feature":      feature_names,
                "Your Input":   [input_vals[f] for f in feature_names],
                "Dataset Mean": mean_vals.values,
            })
            comp_df["Delta %"] = ((comp_df["Your Input"] - comp_df["Dataset Mean"])
                                  / comp_df["Dataset Mean"] * 100).round(1)
            top_delta = comp_df.reindex(
                comp_df["Delta %"].abs().sort_values(ascending=False).index
            ).head(10)
            st.dataframe(
                top_delta.style
                    .background_gradient(subset=["Delta %"], cmap="RdBu_r", vmin=-100, vmax=100)
                    .format({"Your Input": "{:.4f}", "Dataset Mean": "{:.4f}", "Delta %": "{:+.1f}%"}),
                use_container_width=True, height=380
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: BATCH PREDICT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_batch_predict(model, meta):
    st.markdown("""
    <div class='main-header'>
        <h1>📁 Batch Prediction</h1>
        <p>Upload a CSV file and get predictions for all rows</p>
    </div>
    """, unsafe_allow_html=True)

    feature_names = meta["feature_names"]

    st.markdown("""
    <div class='info-box'>
    📌 <b>Instructions:</b><br>
    Upload a <code>.csv</code> file containing the 30 tumour features. 
    Column names must match exactly. The <code>id</code> and <code>diagnosis</code> 
    columns are optional and will be ignored during prediction.
    </div>
    """, unsafe_allow_html=True)

    # ── Sample download ────────────────────────────────────────────────────────
    sample_df = pd.DataFrame([FEATURE_DEFAULTS])
    csv_sample = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Sample CSV Template",
        data=csv_sample,
        file_name="sample_input.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            st.success(f"✅ File loaded: **{uploaded.name}** — {len(raw_df)} rows, {len(raw_df.columns)} columns")

            # ── Validate columns ───────────────────────────────────────────────
            missing_cols = [c for c in feature_names if c not in raw_df.columns]
            extra_cols   = [c for c in raw_df.columns if c not in feature_names
                            and c not in ["id","diagnosis"]]

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()
            if extra_cols:
                st.warning(f"⚠️ Unknown columns will be ignored: {extra_cols}")

            X_upload = raw_df[feature_names].copy()

            # ── Handle nulls ───────────────────────────────────────────────────
            null_count = X_upload.isnull().sum().sum()
            if null_count > 0:
                st.warning(f"⚠️ {null_count} missing values found — imputing with column medians.")
                X_upload.fillna(X_upload.median(), inplace=True)

            with st.spinner(f"Running predictions on {len(X_upload)} samples …"):
                time.sleep(0.3)
                preds = model.predict(X_upload)
                probas = model.predict_proba(X_upload)

            result_df = raw_df.copy()
            result_df["Prediction"]        = ["Malignant" if p == 1 else "Benign" for p in preds]
            result_df["Benign_Prob"]       = probas[:, 0].round(4)
            result_df["Malignant_Prob"]    = probas[:, 1].round(4)
            result_df["Confidence"]        = np.max(probas, axis=1).round(4)

            # ── Summary ─────────────────────────────────────────────────────────
            n_malignant = (preds == 1).sum()
            n_benign    = (preds == 0).sum()
            avg_conf    = np.mean(np.max(probas, axis=1))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='val'>{len(preds)}</div><div class='lbl'>Total Samples</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='val' style='color:#e94560;'>{n_malignant}</div>
                    <div class='lbl'>Malignant</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='val' style='color:#00b894;'>{n_benign}</div>
                    <div class='lbl'>Benign</div></div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class='metric-card'>
                    <div class='val'>{avg_conf*100:.1f}%</div>
                    <div class='lbl'>Avg Confidence</div>
                </div>""", unsafe_allow_html=True)

            # ── Results table ──────────────────────────────────────────────────
            st.markdown("<div class='section-header'>📋 Prediction Results</div>",
                        unsafe_allow_html=True)
            display_cols = (["id"] if "id" in result_df.columns else []) + \
                           ["Prediction","Benign_Prob","Malignant_Prob","Confidence"]
            st.dataframe(
                result_df[display_cols].style
                    .applymap(lambda v: "color:#e94560;font-weight:700" if v=="Malignant"
                              else ("color:#00b894;font-weight:700" if v=="Benign" else ""),
                              subset=["Prediction"])
                    .format({"Benign_Prob":"{:.4f}","Malignant_Prob":"{:.4f}","Confidence":"{:.4f}"}),
                use_container_width=True, height=400
            )

            # ── Download results ───────────────────────────────────────────────
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_out,
                file_name="predictions_output.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: MODEL INSIGHTS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_model_insights(model, meta):
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Model Insights & EDA</h1>
        <p>Explainability, feature analysis, and dataset exploration</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Please place breast-cancer.csv in the app directory.")
        return

    df_work = df.copy()
    le_tmp  = LabelEncoder()
    df_work["diagnosis"] = le_tmp.fit_transform(df_work["diagnosis"])

    feature_names = meta["feature_names"]
    mean_cols     = [c for c in feature_names if c.endswith("_mean")]

    tab1, tab2, tab3 = st.tabs(["🔬 EDA", "🧠 Feature Importance", "🛠️ Model Config"])

    # ── Tab 1: EDA ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-header'>📊 Feature Distributions by Diagnosis</div>",
                    unsafe_allow_html=True)

        selected_feat = st.selectbox("Select feature to explore:", mean_cols)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#0f3460")

        benign_vals    = df_work[df_work["diagnosis"]==0][selected_feat]
        malignant_vals = df_work[df_work["diagnosis"]==1][selected_feat]

        axes[0].hist(benign_vals,    bins=25, color="#00b894", alpha=0.75,
                     edgecolor="none", label=f"Benign (n={len(benign_vals)})")
        axes[0].hist(malignant_vals, bins=25, color="#e94560", alpha=0.75,
                     edgecolor="none", label=f"Malignant (n={len(malignant_vals)})")
        axes[0].set_title(f"{selected_feat} Distribution",
                          color="white", fontsize=11, fontweight="bold")
        axes[0].legend(facecolor="#0f3460", labelcolor="white")
        axes[0].set_ylabel("Count", color="white")

        axes[1].boxplot([benign_vals, malignant_vals], labels=["Benign","Malignant"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#1a1a2e"),
                        medianprops=dict(color="#f9ca24", linewidth=2.5),
                        whiskerprops=dict(color="white"),
                        capprops=dict(color="white"),
                        flierprops=dict(marker="o", color="#e94560",
                                        markersize=4, alpha=0.5))
        for patch, color in zip(axes[1].patches, ["#00b894","#e94560"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        axes[1].set_title(f"{selected_feat} Box Plot",
                          color="white", fontsize=11, fontweight="bold")
        axes[1].tick_params(colors="white")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Correlation heatmap
        st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(13, 9))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        corr = df_work[mean_cols + ["diagnosis"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, linewidths=0.5,
                    annot_kws={"size":8, "color":"white"},
                    ax=ax)
        ax.set_title("Correlation — Mean Features + Diagnosis",
                     color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tab 2: Feature Importance ───────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class='info-box'>
        🧠 <b>Permutation Importance:</b> Measures how much the ROC-AUC score 
        drops when a feature is randomly shuffled — a larger drop = more important feature.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Computing permutation importances …"):
            X_full = df_work[feature_names]
            y_full = df_work["diagnosis"]
            _, X_test_fi, _, y_test_fi = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
            result = permutation_importance(
                model, X_test_fi, y_test_fi,
                n_repeats=15, random_state=42,
                scoring="roc_auc", n_jobs=-1)

        perm_df = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": result.importances_mean,
            "Std":        result.importances_std
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        colors = ["#e94560" if v > 0 else "#555" for v in perm_df["Importance"]]
        ax.barh(perm_df["Feature"], perm_df["Importance"],
                xerr=perm_df["Std"], color=colors,
                edgecolor="none", capsize=3, alpha=0.85)
        ax.set_title("Feature Importances (Permutation — ROC-AUC Drop)",
                     color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Decrease in ROC-AUC", color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#0f3460")
        ax.axvline(0, color="#555", linestyle="-", lw=0.8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<div class='section-header'>🏆 Top 10 Features</div>",
                    unsafe_allow_html=True)
        top10 = perm_df.sort_values("Importance", ascending=False).head(10)
        st.dataframe(
            top10[["Feature","Importance","Std"]].style
                .background_gradient(subset=["Importance"], cmap="RdBu_r")
                .format({"Importance":"{:.4f}", "Std":"{:.4f}"}),
            use_container_width=True
        )

    # ── Tab 3: Model Config ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-header'>🛠️ Model Configuration</div>",
                    unsafe_allow_html=True)
        st.json({
            "model_type": "SVC (Support Vector Classifier)",
            "pipeline_steps": ["StandardScaler", "SVC"],
            "best_hyperparameters": meta.get("best_params", {}),
            "cv_strategy": "StratifiedKFold(n_splits=5, shuffle=True)",
            "scoring": "roc_auc",
            "n_features": len(feature_names),
            "test_accuracy": f"{meta['test_accuracy']*100:.2f}%",
            "test_roc_auc": f"{meta['test_roc_auc']:.4f}",
        })

        st.markdown("<div class='section-header'>📦 Feature Names</div>",
                    unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            "Index": range(len(feature_names)),
            "Feature": feature_names,
            "Group": ["Mean" if f.endswith("_mean")
                      else "SE" if f.endswith("_se")
                      else "Worst" for f in feature_names]
        })
        st.dataframe(feat_df, use_container_width=True, height=350)

        # Model files
        st.markdown("<div class='section-header'>💾 Saved Model Files</div>",
                    unsafe_allow_html=True)
        for path in [MODEL_PATH, META_PATH]:
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024
                st.success(f"✅ `{path}` — {size:.1f} KB")
            else:
                st.warning(f"⚠️ `{path}` — not found (will be auto-created on first run)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN ENTRYPOINT                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def main():
    # ── Load model ─────────────────────────────────────────────────────────────
    with st.spinner("Loading model …"):
        model, meta, trained_fresh = load_or_train_model()

    if model is None:
        st.error("❌ Model could not be loaded and `breast-cancer.csv` was not found. "
                 "Please place both files in the same directory as `app.py`.")
        st.stop()

    if trained_fresh:
        st.success("✅ Model trained and saved successfully!")

    # ── Render sidebar + route ─────────────────────────────────────────────────
    page = render_sidebar()

    if   "Home"     in page: page_home(model, meta)
    elif "Manual"   in page: page_manual_predict(model, meta)
    elif "Batch"    in page: page_batch_predict(model, meta)
    elif "Insights" in page: page_model_insights(model, meta)


if __name__ == "__main__":
    main()
