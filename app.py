import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ModelCheck AI", page_icon="🧪", layout="wide")
st.title("🧪 ModelCheck AI")
st.write(
    "Compare machine-learning algorithms under Normal, Noise and "
    "Feature-Shift conditions. The app detects the task type and explains results."
)


# -----------------------------------------------------------------------------
# 1. AUTOMATIC TARGET-COLUMN AND TASK DETECTION
# -----------------------------------------------------------------------------

TARGET_NAMES = [
    "target", "label", "class", "category", "species", "outcome",
    "diagnosis", "result", "status", "survived", "churn", "approved",
    "price", "salary", "score", "performance", "temperature", "amount",
    "revenue", "sales", "weight", "height", "age", "energy", "power",
    "voltage", "current", "distance", "measurement", "profit", "cost",
    "income", "duration", "y"
]


def suggest_target_column(frame):
    """Return a likely target column; use last column as a transparent fallback."""
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    for name in TARGET_NAMES:
        if name in normalized:
            return normalized[name], "Matched a commonly used target-column name."
    # Prefer a column with a small number of distinct values if it is near the end.
    last = frame.columns[-1]
    return last, "No obvious target name found; using the last column as a suggestion. Please verify it."


def detect_task(y, target_name, choice="Automatic"):
    if choice in ("Classification", "Regression"):
        return choice, "Task type selected manually."

    values = y.dropna()
    if values.empty:
        raise ValueError("The selected target column has no valid values.")

    name = str(target_name).strip().lower()
    classification_words = [
        "class", "category", "species", "label", "outcome", "diagnosis",
        "type", "status", "churn", "survived", "approved", "segment"
    ]
    regression_words = [
        "price", "salary", "score", "performance", "temperature", "amount",
        "revenue", "sales", "weight", "height", "age", "energy", "power",
        "voltage", "current", "distance", "measurement", "profit", "cost",
        "income", "duration"
    ]
    if any(word in name for word in classification_words):
        return "Classification", "The target-column name suggests class labels or categories."
    if any(word in name for word in regression_words):
        return "Regression", "The target-column name suggests a measurable numeric value."

    if (pd.api.types.is_object_dtype(values) or
            isinstance(values.dtype, pd.CategoricalDtype) or
            pd.api.types.is_bool_dtype(values)):
        return "Classification", "The target contains text, category or Boolean values."

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "Classification", "The target values are non-numeric categories."

    unique_count = numeric.nunique()
    unique_ratio = unique_count / max(len(numeric), 1)
    if unique_count == 2:
        return "Classification", "The numeric target has exactly two distinct classes."
    if pd.api.types.is_integer_dtype(numeric) and unique_count <= 10:
        return "Classification", "The integer target has a small number of distinct values; treated as class labels."
    if unique_count > 10 or unique_ratio > 0.05:
        return "Regression", "The numeric target has many distinct values; treated as a continuous quantity."
    return "Classification", "The numeric target has relatively few distinct values; treated as class labels."


# -----------------------------------------------------------------------------
# 2. ALGORITHMS
# -----------------------------------------------------------------------------

def get_models(task):
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=3000),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        }
    return {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    }


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def evaluate(task, actual, predicted):
    if task == "Classification":
        return {
            "Accuracy": accuracy_score(actual, predicted),
            "Balanced Accuracy": balanced_accuracy_score(actual, predicted),
            "Precision (weighted)": precision_score(actual, predicted, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(actual, predicted, average="weighted", zero_division=0),
            "F1 Score (weighted)": f1_score(actual, predicted, average="weighted", zero_division=0),
        }
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R² Score": r2_score(actual, predicted) if len(actual) > 1 else np.nan,
    }


# -----------------------------------------------------------------------------
# 3. TEST-DATA PERTURBATIONS
# -----------------------------------------------------------------------------

def create_noise(X_test, X_train, numeric_cols, strength, seed):
    result = X_test.copy()
    rng = np.random.default_rng(seed)
    for col in numeric_cols:
        values = pd.to_numeric(result[col], errors="coerce").astype(float).to_numpy(copy=True)
        train_values = pd.to_numeric(X_train[col], errors="coerce").astype(float).to_numpy()
        train_values = train_values[np.isfinite(train_values)]
        scale = np.std(train_values) if len(train_values) else 0.0
        valid = np.isfinite(values)
        values[valid] += rng.normal(0, strength * scale, valid.sum())
        result[col] = values
    return result


def create_shift(X_test, X_train, feature, strength):
    result = X_test.copy()
    values = pd.to_numeric(result[feature], errors="coerce").astype(float).to_numpy(copy=True)
    train_values = pd.to_numeric(X_train[feature], errors="coerce").astype(float).to_numpy()
    train_values = train_values[np.isfinite(train_values)]
    scale = np.std(train_values) if len(train_values) else 0.0
    valid = np.isfinite(values)
    values[valid] += strength * scale
    result[feature] = values
    return result


# -----------------------------------------------------------------------------
# 4. UPLOAD AND VALIDATE DATASET
# -----------------------------------------------------------------------------
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to start.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

df = df.dropna(axis=1, how="all").drop_duplicates()
if len(df) < 5 or len(df.columns) < 2:
    st.error("Dataset needs at least 5 rows and 2 columns.")
    st.stop()

st.subheader("Dataset Preview")
st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
st.dataframe(df.head(15), use_container_width=True)

suggested_target, target_reason = suggest_target_column(df)
st.subheader("Target Column and Task Detection")
st.caption("The app suggests a target column automatically. Check that it is the value you want the model to predict.")
cols = list(df.columns)
selected_index = cols.index(suggested_target)
target = st.selectbox("Target column to predict", cols, index=selected_index)
st.caption(f"Target suggestion: {target_reason}")

task_choice = st.selectbox("Task type", ["Automatic", "Classification", "Regression"], index=0)
try:
    task, task_reason = detect_task(df[target], target, task_choice)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
st.success(f"Detected task: {task}")
st.caption(task_reason)

work = df.dropna(subset=[target]).copy()
y = work[target].copy()
X = work.drop(columns=[target]).copy()

# Remove common identifier columns, which can mislead models.
drop_cols = [c for c in X.columns if str(c).strip().lower() in [
    "id", "index", "unnamed: 0", "source_file", "filename"
]]
X = X.drop(columns=drop_cols, errors="ignore")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
valid_rows = ~X.isna().all(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

if X.empty or X.shape[1] == 0:
    st.error("No usable feature columns remain after removing empty/identifier columns.")
    st.stop()

if task == "Classification":
    y = y.astype(str)
    if y.nunique() < 2:
        st.error("Classification requires at least two target classes.")
        st.stop()
else:
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(float)
    if y.nunique() < 2:
        st.error("Regression requires at least two different numeric target values.")
        st.stop()

if len(X) < 5:
    st.error("Not enough usable rows remain after cleaning the selected target and features.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. AUTO OR MANUAL ALGORITHM SELECTION
# -----------------------------------------------------------------------------
st.subheader("Algorithm Selection")
models = get_models(task)
selection_mode = st.radio(
    "How should algorithms be selected?",
    ["Automatic", "Manual"], horizontal=True,
    help="Automatic selects all four suitable algorithms for the detected task. Manual lets you choose a subset."
)
if selection_mode == "Automatic":
    selected_models = list(models.keys())
    st.info("Automatically selected suitable algorithms: " + ", ".join(selected_models))
else:
    selected_models = st.multiselect(
        "Choose algorithms to compare", options=list(models.keys()), default=list(models.keys())
    )
    if not selected_models:
        st.warning("Select at least one algorithm to continue.")
        st.stop()

# -----------------------------------------------------------------------------
# 6. EXPERIMENT SETTINGS
# -----------------------------------------------------------------------------
st.subheader("Configure Test Conditions")
col1, col2, col3 = st.columns(3)
with col1:
    run_normal = st.checkbox("Normal", value=True)
with col2:
    run_noise = st.checkbox("Noise", value=True)
with col3:
    run_shift = st.checkbox("Feature-Shift", value=True)

noise_strength = st.slider("Noise strength", 0.0, 1.0, 0.2, 0.05)
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
shift_feature = None
shift_strength = 1.0
if numeric_cols:
    shift_feature = st.selectbox("Feature to shift", numeric_cols)
    shift_strength = st.slider("Feature-Shift strength (standard deviations)", 0.0, 3.0, 1.0, 0.25)
else:
    st.warning("No numeric input features found; Noise and Feature-Shift cannot be applied.")
    run_noise = False
    run_shift = False

test_fraction = st.slider("Test set percentage", 20, 40, 20, 5) / 100
seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
conditions = []
if run_normal:
    conditions.append("Normal")
if run_noise and numeric_cols:
    conditions.append("Noise")
if run_shift and shift_feature is not None:
    conditions.append("Feature-Shift")
if not conditions:
    st.warning("Select at least one test condition.")
    st.stop()

# -----------------------------------------------------------------------------
# 7. RUN EXPERIMENT
# -----------------------------------------------------------------------------
if st.button("Run Model Comparison", type="primary"):
    try:
        stratify = None
        if task == "Classification":
            counts = y.value_counts()
            # Stratify only when each class can appear in both train and test.
            if len(counts) > 1 and counts.min() >= 2:
                stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=int(seed), stratify=stratify
        )
        if task == "Classification" and y_train.nunique() < 2:
            st.error("Training data contains only one class. Upload more data or change the split.")
            st.stop()

        numeric_train = X_train.select_dtypes(include=np.number).columns.tolist()
        test_sets = {}
        if run_normal:
            test_sets["Normal"] = X_test.copy()
        if run_noise and numeric_train:
            test_sets["Noise"] = create_noise(X_test, X_train, numeric_train, noise_strength, int(seed))
        if run_shift and shift_feature is not None:
            test_sets["Feature-Shift"] = create_shift(X_test, X_train, shift_feature, shift_strength)

        metric_rows, prediction_frames, errors = [], [], []
        progress = st.progress(0)
        for index, name in enumerate(selected_models):
            try:
                pipeline = Pipeline([
                    ("prep", make_preprocessor(X_train)),
                    ("model", clone(models[name])),
                ])
                pipeline.fit(X_train, y_train)
                for condition_name, X_condition in test_sets.items():
                    predictions = pipeline.predict(X_condition)
                    metric_values = evaluate(task, y_test, predictions)
                    metric_rows.append({
                        "Algorithm": name, "Condition": condition_name, **metric_values
                    })
                    prediction_frames.append(pd.DataFrame({
                        "Row_Index": X_condition.index,
                        "Algorithm": name,
                        "Condition": condition_name,
                        "Actual": np.asarray(y_test),
                        "Predicted": np.asarray(predictions),
                    }))
            except Exception as exc:
                errors.append({"Algorithm": name, "Error": f"{type(exc).__name__}: {exc}"})
            progress.progress((index + 1) / len(selected_models))
        progress.empty()

        if not metric_rows:
            st.error("All algorithms failed. Review the error table below.")
            if errors:
                st.dataframe(pd.DataFrame(errors), use_container_width=True)
            st.stop()
        st.session_state["mc_results"] = {
            "task": task,
            "metrics": pd.DataFrame(metric_rows),
            "predictions": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
            "errors": pd.DataFrame(errors),
            "selected_models": selected_models,
            "target": target,
        }
    except Exception as exc:
        st.error(f"Experiment failed: {type(exc).__name__}: {exc}")
        st.stop()

# -----------------------------------------------------------------------------
# 8. RESULTS, GRAPH AND EXPLICIT SVM/SVR VALUE DISPLAY
# -----------------------------------------------------------------------------
if "mc_results" not in st.session_state:
    st.info("Configure the settings and click Run Model Comparison.")
    st.stop()

results = st.session_state["mc_results"]
metrics_df = results["metrics"].copy()
predictions_df = results["predictions"]
errors_df = results["errors"]
result_task = results["task"]
result_models = results["selected_models"]

st.subheader("Model Comparison Results")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)
if not errors_df.empty:
    st.warning("Some algorithms could not run. Failed models are not assigned fake zero scores.")
    st.dataframe(errors_df, use_container_width=True, hide_index=True)

st.subheader("Normal vs Noise vs Feature-Shift")
metric_options = (
    ["Accuracy", "Balanced Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 Score (weighted)"]
    if result_task == "Classification" else ["RMSE", "MAE", "R² Score"]
)
available_metrics = [m for m in metric_options if m in metrics_df.columns and metrics_df[m].notna().any()]
if not available_metrics:
    st.warning("No valid metric values are available to plot.")
    graph_metric = None
else:
    graph_metric = st.selectbox("Select graph metric", available_metrics)

    # Reindex ensures selected algorithms remain visible and values are not silently dropped.
    pivot = metrics_df.pivot_table(index="Algorithm", columns="Condition", values=graph_metric, aggfunc="mean")
    pivot = pivot.reindex([m for m in result_models if m in pivot.index])
    condition_order = [c for c in ["Normal", "Noise", "Feature-Shift"] if c in pivot.columns]
    pivot = pivot.reindex(columns=condition_order)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if not pivot.empty and pivot.notna().any().any():
        pivot.plot(kind="bar", ax=ax, width=0.78)
        ax.set_title(f"{graph_metric}: Algorithm Performance by Test Condition")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(graph_metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Condition")
        ax.tick_params(axis="x", rotation=20)
        # Annotate every finite bar so SVM/SVR results are visible and readable.
        for container in ax.containers:
            labels = []
            for bar in container:
                height = bar.get_height()
                labels.append(f"{height:.3f}" if np.isfinite(height) else "")
            ax.bar_label(container, labels=labels, padding=2, fontsize=8)
        ax.margins(y=0.15)
    else:
        ax.text(0.5, 0.5, "No finite metric values available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    graph_bytes = io.BytesIO()
    fig.savefig(graph_bytes, format="png", dpi=180, bbox_inches="tight")
    graph_bytes.seek(0)
    plt.close(fig)

    # Explicit SVM/SVR result card/table prevents users from overlooking its numeric value.
    svm_names = [n for n in ["SVM", "SVR"] if n in metrics_df["Algorithm"].unique()]
    if svm_names:
        st.markdown("**SVM / SVR metric values**")
        svm_values = metrics_df[metrics_df["Algorithm"].isin(svm_names)][["Algorithm", "Condition", graph_metric]]
        if not svm_values.empty:
            st.dataframe(svm_values, use_container_width=True, hide_index=True)
        else:
            st.info("SVM/SVR was selected but did not produce a metric value. Check the algorithm error details above.")

# -----------------------------------------------------------------------------
# 9. RELIABILITY ANALYSIS
# -----------------------------------------------------------------------------
st.subheader("Reliability Analysis")
changes = []
if "Normal" in metrics_df["Condition"].values:
    normal = metrics_df[metrics_df["Condition"] == "Normal"].set_index("Algorithm")
    for _, row in metrics_df.iterrows():
        if row["Condition"] == "Normal" or row["Algorithm"] not in normal.index:
            continue
        for metric in available_metrics:
            normal_value = normal.loc[row["Algorithm"], metric]
            if isinstance(normal_value, pd.Series):
                normal_value = normal_value.iloc[0]
            if pd.notna(normal_value) and pd.notna(row[metric]):
                changes.append({
                    "Algorithm": row["Algorithm"],
                    "Condition": row["Condition"],
                    "Metric": metric,
                    "Normal value": normal_value,
                    "Condition value": row[metric],
                    "Change": row[metric] - normal_value,
                })
changes_df = pd.DataFrame(changes)
if not changes_df.empty:
    st.dataframe(changes_df, use_container_width=True, hide_index=True)
else:
    st.info("Enable Normal and at least one perturbed condition to compare robustness against the baseline.")

# -----------------------------------------------------------------------------
# 10. AUTOMATIC MODEL INSIGHTS AND CONCLUSION
# -----------------------------------------------------------------------------
st.subheader("Model Insights & Conclusion")
st.write("This section interprets the actual metrics generated in this run. It does not use fixed sample results.")

# Higher is better for these metrics; lower is better for error metrics.
higher_is_better = result_task == "Classification" or graph_metric == "R² Score"
primary_metric = graph_metric if graph_metric else ("Accuracy" if result_task == "Classification" else "RMSE")

normal_rows = metrics_df[metrics_df["Condition"] == "Normal"].dropna(subset=[primary_metric]) if primary_metric in metrics_df else pd.DataFrame()
if not normal_rows.empty:
    best_idx = normal_rows[primary_metric].idxmax() if higher_is_better else normal_rows[primary_metric].idxmin()
    best_row = normal_rows.loc[best_idx]
    st.markdown(f"**Best model on Normal test data:** {best_row['Algorithm']} ({primary_metric} = {best_row[primary_metric]:.4f}).")
else:
    st.markdown("**Normal-condition comparison:** A Normal baseline was not run, so a best Normal-condition model cannot be identified.")

if not normal_rows.empty and len(normal_rows) > 1:
    values = normal_rows.set_index("Algorithm")[primary_metric]
    spread = float(values.max() - values.min())
    st.write(f"The difference between the highest and lowest Normal-condition {primary_metric} is {spread:.4f}.")

if "Normal" in metrics_df["Condition"].values and not changes_df.empty:
    for condition in ["Noise", "Feature-Shift"]:
        subset = changes_df[changes_df["Condition"] == condition]
        subset = subset[subset["Metric"] == primary_metric]
        if subset.empty:
            continue
        # For higher-is-better metrics, negative change is a deterioration.
        subset = subset.copy()
        subset["Performance drop"] = -subset["Change"] if higher_is_better else subset["Change"]
        most_robust = subset.loc[subset["Performance drop"].idxmin()]
        most_affected = subset.loc[subset["Performance drop"].idxmax()]
        st.write(
            f"**{condition}:** {most_robust['Algorithm']} had the smallest change from Normal "
            f"({most_robust['Change']:+.4f}), while {most_affected['Algorithm']} had the largest "
            f"change ({most_affected['Change']:+.4f}) in {primary_metric}."
        )

# A readable overall conclusion, grounded in the run's measured results.
if not normal_rows.empty:
    best_name = str(best_row["Algorithm"])
    conclusion = (
        f"In this experiment, {best_name} achieved the strongest Normal-condition result "
        f"according to {primary_metric}. However, the best Normal score alone does not establish "
        f"that a model is the most reliable under changed data. Compare its Noise and Feature-Shift "
        f"results with the Normal baseline to see whether performance is maintained or drops. "
    )
else:
    conclusion = (
        "This run does not include a Normal baseline, so it cannot determine which model performs "
        "best on the original test data or quantify degradation from that baseline. "
    )

if changes_df.empty:
    conclusion += "Run Normal together with Noise and/or Feature-Shift to obtain a direct robustness comparison."
else:
    condition_count = changes_df["Condition"].nunique()
    conclusion += (
        f"The reliability table compares model metrics across {condition_count} perturbed condition(s). "
        "A smaller deterioration from the Normal result indicates better stability for the tested "
        "perturbation; it does not guarantee performance on every real-world dataset."
    )

st.info(conclusion)
st.caption(
    "Interpretation note: Noise and Feature-Shift are synthetic test transformations. "
    "These results describe this dataset, split, selected features and settings only. "
    "They do not prove that a model will behave the same way in every deployment environment."
)

# -----------------------------------------------------------------------------
# 11. PREDICTIONS AND DOWNLOADS
# -----------------------------------------------------------------------------
st.subheader("Predictions")
normal_predictions = predictions_df[predictions_df["Condition"] == "Normal"].copy() if not predictions_df.empty else pd.DataFrame()
with st.expander("Normal-condition predictions", expanded=False):
    if not normal_predictions.empty:
        st.dataframe(normal_predictions, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Normal Predictions CSV",
            normal_predictions.to_csv(index=False).encode("utf-8"),
            "modelcheck_normal_predictions.csv", "text/csv", key="download_normal_predictions"
        )
    else:
        st.warning("Normal predictions are unavailable. Enable Normal and rerun the experiment.")
with st.expander("All-condition predictions", expanded=False):
    if not predictions_df.empty:
        st.dataframe(predictions_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No predictions are available.")

st.subheader("Download Results")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False).encode("utf-8"), "modelcheck_metrics.csv", "text/csv", key="download_metrics")
with col2:
    st.download_button("Download All Predictions CSV", predictions_df.to_csv(index=False).encode("utf-8"), "modelcheck_predictions.csv", "text/csv", key="download_all_predictions")
if not changes_df.empty:
    st.download_button("Download Reliability Analysis CSV", changes_df.to_csv(index=False).encode("utf-8"), "modelcheck_reliability_changes.csv", "text/csv", key="download_reliability")
if graph_metric:
    st.download_button("Download Comparison Graph PNG", graph_bytes.getvalue(), "modelcheck_comparison_graph.png", "image/png", key="download_graph")
if not errors_df.empty:
    st.download_button("Download Algorithm Errors CSV", errors_df.to_csv(index=False).encode("utf-8"), "modelcheck_errors.csv", "text/csv", key="download_errors")

st.caption(
    "Normal evaluates the original held-out test data. Noise adds random variation to numeric "
    "features. Feature-Shift offsets one selected numeric feature. Algorithm selection is "
    "automatic by default; manual selection is also available."
)
