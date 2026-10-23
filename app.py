
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

st.set_page_config(
    page_title="ModelCheck AI",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 ModelCheck AI")
st.write(
    "Compare four machine-learning algorithms under "
    "Normal, Noise and Feature-Shift conditions."
)


# --------------------------------------------------
# 1. TASK DETECTION
# --------------------------------------------------

def detect_task(y, target_name, choice):
    if choice != "Automatic":
        return choice

    name = str(target_name).lower()

    regression_words = [
        "price", "salary", "score", "performance",
        "temperature", "amount", "revenue", "sales",
        "weight", "height", "age", "energy", "power",
        "voltage", "current", "distance", "measurement"
    ]

    if any(word in name for word in regression_words):
        return "Regression"

    if (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
    ):
        return "Classification"

    values = pd.to_numeric(y, errors="coerce").dropna()

    if values.nunique() == 2 and set(values.unique()).issubset({0, 1}):
        return "Classification"

    if (
        pd.api.types.is_integer_dtype(values)
        and values.nunique() <= 10
        and values.nunique() / max(len(values), 1) <= 0.05
    ):
        return "Classification"

    return "Regression"


# --------------------------------------------------
# 2. ALGORITHMS
# --------------------------------------------------

def get_models(task):
    if task == "Classification":
        return {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42
            ),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "SVM": SVC()
        }

    return {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "SVR": SVR()
    }


# --------------------------------------------------
# 3. PREPROCESSING
# --------------------------------------------------

def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []

    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    if categorical:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore", sparse=False
            )

        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder)
            ]),
            categorical
        ))

    return ColumnTransformer(transformers, remainder="drop")


# --------------------------------------------------
# 4. METRICS
# --------------------------------------------------

def evaluate(task, actual, predicted):
    if task == "Classification":
        return {
            "Accuracy": accuracy_score(actual, predicted),
            "Balanced Accuracy": balanced_accuracy_score(actual, predicted),
            "Precision (weighted)": precision_score(
                actual, predicted, average="weighted", zero_division=0
            ),
            "Recall (weighted)": recall_score(
                actual, predicted, average="weighted", zero_division=0
            ),
            "F1 Score (weighted)": f1_score(
                actual, predicted, average="weighted", zero_division=0
            )
        }

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R² Score": (
            r2_score(actual, predicted)
            if len(actual) > 1 else np.nan
        )
    }


# --------------------------------------------------
# 5. PERTURBATION CONDITIONS
# --------------------------------------------------

def create_noise(X_test, X_train, numeric_cols, strength, seed):
    result = X_test.copy()
    rng = np.random.default_rng(seed)

    for col in numeric_cols:
        values = pd.to_numeric(
            result[col], errors="coerce"
        ).astype(float).to_numpy(copy=True)

        train_values = pd.to_numeric(
            X_train[col], errors="coerce"
        ).astype(float).to_numpy()

        train_values = train_values[np.isfinite(train_values)]
        scale = np.std(train_values) if len(train_values) else 0.0

        valid = np.isfinite(values)
        values[valid] += rng.normal(
            0, strength * scale, valid.sum()
        )

        result[col] = values

    return result


def create_shift(X_test, X_train, feature, strength):
    result = X_test.copy()

    values = pd.to_numeric(
        result[feature], errors="coerce"
    ).astype(float).to_numpy(copy=True)

    train_values = pd.to_numeric(
        X_train[feature], errors="coerce"
    ).astype(float).to_numpy()

    train_values = train_values[np.isfinite(train_values)]
    scale = np.std(train_values) if len(train_values) else 0.0

    valid = np.isfinite(values)
    values[valid] += strength * scale

    result[feature] = values
    return result


# --------------------------------------------------
# 6. UPLOAD DATASET
# --------------------------------------------------

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


# --------------------------------------------------
# 7. TARGET AND TASK
# --------------------------------------------------

st.subheader("Select Target and Task")

target = st.selectbox(
    "Target column to predict",
    list(df.columns),
    index=len(df.columns) - 1
)

task_choice = st.selectbox(
    "Task type",
    ["Automatic", "Classification", "Regression"]
)

task = detect_task(df[target], target, task_choice)

st.success(f"Task used: {task}")

work = df.dropna(subset=[target]).copy()
y = work[target].copy()
X = work.drop(columns=[target]).copy()

# Remove likely identifier columns
drop_cols = [
    c for c in X.columns
    if str(c).strip().lower() in [
        "id", "index", "unnamed: 0", "source_file", "filename"
    ]
]
X = X.drop(columns=drop_cols, errors="ignore")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

valid_rows = ~X.isna().all(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

if X.empty or X.shape[1] == 0:
    st.error("No usable feature columns remain.")
    st.stop()

if task == "Classification":
    y = y.astype(str)

    if y.nunique() < 2:
        st.error("Classification needs at least two classes.")
        st.stop()
else:
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(float)

    if y.nunique() < 2:
        st.error("Regression needs at least two different numeric target values.")
        st.stop()


# --------------------------------------------------
# 8. ALGORITHM SELECTION
# --------------------------------------------------

st.subheader("Select Algorithms")

models = get_models(task)

selected_models = st.multiselect(
    "Choose algorithms to compare",
    options=list(models.keys()),
    default=list(models.keys())
)

if not selected_models:
    st.warning("Select at least one algorithm.")
    st.stop()


# --------------------------------------------------
# 9. NORMAL, NOISE AND FEATURE-SHIFT OPTIONS
# --------------------------------------------------

st.subheader("Configure Test Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    run_normal = st.checkbox("Normal", value=True)

with col2:
    run_noise = st.checkbox("Noise", value=True)

with col3:
    run_shift = st.checkbox("Feature-Shift", value=True)

noise_strength = st.slider(
    "Noise strength",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

shift_feature = None
shift_strength = 1.0

if numeric_cols:
    shift_feature = st.selectbox(
        "Feature to shift", numeric_cols
    )

    shift_strength = st.slider(
        "Feature-Shift strength (standard deviations)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.25
    )
else:
    st.warning(
        "No numeric input features found; Noise and Feature-Shift "
        "cannot be applied to this dataset."
    )
    run_noise = False
    run_shift = False

test_fraction = st.slider(
    "Test set percentage", 20, 40, 20, 5
) / 100

seed = st.number_input(
    "Random seed", min_value=0, max_value=999999, value=42
)

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


# --------------------------------------------------
# 10. RUN EXPERIMENT
# --------------------------------------------------

if st.button("Run Model Comparison", type="primary"):
    try:
        stratify = None

        if task == "Classification":
            counts = y.value_counts()
            if len(counts) > 1 and counts.min() >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_fraction,
            random_state=int(seed),
            stratify=stratify
        )

        if task == "Classification" and y_train.nunique() < 2:
            st.error("Training data contains only one class. Upload more data.")
            st.stop()

        numeric_train = X_train.select_dtypes(
            include=np.number
        ).columns.tolist()

        test_sets = {}

        if run_normal:
            test_sets["Normal"] = X_test.copy()

        if run_noise and numeric_train:
            test_sets["Noise"] = create_noise(
                X_test, X_train, numeric_train,
                noise_strength, int(seed)
            )

        if run_shift and shift_feature is not None:
            test_sets["Feature-Shift"] = create_shift(
                X_test, X_train, shift_feature, shift_strength
            )

        metric_rows = []
        prediction_frames = []
        errors = []

        progress = st.progress(0)

        for index, name in enumerate(selected_models):
            try:
                pipeline = Pipeline([
                    ("prep", make_preprocessor(X_train)),
                    ("model", clone(models[name]))
                ])

                pipeline.fit(X_train, y_train)

                for condition_name, X_condition in test_sets.items():
                    predictions = pipeline.predict(X_condition)

                    metric_rows.append({
                        "Algorithm": name,
                        "Condition": condition_name,
                        **evaluate(task, y_test, predictions)
                    })

                    prediction_frames.append(pd.DataFrame({
                        "Row_Index": X_condition.index,
                        "Algorithm": name,
                        "Condition": condition_name,
                        "Actual": np.asarray(y_test),
                        "Predicted": np.asarray(predictions)
                    }))

            except Exception as exc:
                errors.append({
                    "Algorithm": name,
                    "Error": f"{type(exc).__name__}: {exc}"
                })

            progress.progress((index + 1) / len(selected_models))

        progress.empty()

        if not metric_rows:
            st.error("All algorithms failed. Check the error table.")
            if errors:
                st.dataframe(pd.DataFrame(errors))
            st.stop()

        st.session_state["mc_results"] = {
            "task": task,
            "metrics": pd.DataFrame(metric_rows),
            "predictions": (
                pd.concat(prediction_frames, ignore_index=True)
                if prediction_frames else pd.DataFrame()
            ),
            "errors": pd.DataFrame(errors)
        }

    except Exception as exc:
        st.error(f"Experiment failed: {type(exc).__name__}: {exc}")
        st.stop()


# --------------------------------------------------
# 11. RESULTS AND GRAPHS
# --------------------------------------------------

if "mc_results" not in st.session_state:
    st.info("Configure the settings and click Run Model Comparison.")
    st.stop()

results = st.session_state["mc_results"]
metrics_df = results["metrics"]
predictions_df = results["predictions"]
errors_df = results["errors"]
result_task = results["task"]

st.subheader("Model Comparison Results")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

if not errors_df.empty:
    st.warning("Some algorithms failed; no fake zero scores are assigned.")
    st.dataframe(errors_df, use_container_width=True, hide_index=True)

st.subheader("Normal vs Noise vs Feature-Shift")

if result_task == "Classification":
    metric_options = [
        "Accuracy", "Balanced Accuracy", "Precision (weighted)",
        "Recall (weighted)", "F1 Score (weighted)"
    ]
else:
    metric_options = ["RMSE", "MAE", "R² Score"]

available_metrics = [
    m for m in metric_options if m in metrics_df.columns
]

graph_metric = st.selectbox(
    "Select graph metric",
    available_metrics
)

pivot = metrics_df.pivot(
    index="Algorithm",
    columns="Condition",
    values=graph_metric
)

order = [
    c for c in ["Normal", "Noise", "Feature-Shift"]
    if c in pivot.columns
]
pivot = pivot[order]

fig, ax = plt.subplots(figsize=(10, 5))
pivot.plot(kind="bar", ax=ax)
ax.set_title(f"{graph_metric}: Algorithm and Test Condition")
ax.set_xlabel("Algorithm")
ax.set_ylabel(graph_metric)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Condition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

st.pyplot(fig, use_container_width=True)

graph_bytes = io.BytesIO()
fig.savefig(graph_bytes, format="png", dpi=180, bbox_inches="tight")
graph_bytes.seek(0)
plt.close(fig)


# --------------------------------------------------
# 12. RELIABILITY ANALYSIS
# --------------------------------------------------

st.subheader("Reliability Analysis")

changes = []

if "Normal" in metrics_df["Condition"].values:
    normal = metrics_df[
        metrics_df["Condition"] == "Normal"
    ].set_index("Algorithm")

    for _, row in metrics_df.iterrows():
        if row["Condition"] == "Normal":
            continue

        for metric in available_metrics:
            normal_value = normal.loc[row["Algorithm"], metric]

            changes.append({
                "Algorithm": row["Algorithm"],
                "Condition": row["Condition"],
                "Metric": metric,
                "Normal value": normal_value,
                "Condition value": row[metric],
                "Change": row[metric] - normal_value
            })

changes_df = pd.DataFrame(changes)

if not changes_df.empty:
    st.dataframe(changes_df, use_container_width=True, hide_index=True)
else:
    st.info("Enable Normal to compare results against the baseline.")


# --------------------------------------------------
# 13. PREDICTIONS AND DOWNLOADS
# --------------------------------------------------

st.subheader("Predictions")

with st.expander("Normal-condition predictions"):
    st.dataframe(
        predictions_df[predictions_df["Condition"] == "Normal"],
        use_container_width=True,
        hide_index=True
    )

with st.expander("All-condition predictions"):
    st.dataframe(
        predictions_df,
        use_container_width=True,
        hide_index=True
    )

st.subheader("Download Results")

st.download_button(
    "Download Metrics CSV",
    metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="modelcheck_metrics.csv",
    mime="text/csv"
)

st.download_button(
    "Download Predictions CSV",
    predictions_df.to_csv(index=False).encode("utf-8"),
    file_name="modelcheck_predictions.csv",
    mime="text/csv"
)

if not changes_df.empty:
    st.download_button(
        "Download Reliability Changes CSV",
        changes_df.to_csv(index=False).encode("utf-8"),
        file_name="modelcheck_reliability_changes.csv",
        mime="text/csv"
    )

st.download_button(
    "Download Comparison Graph PNG",
    graph_bytes.getvalue(),
    file_name="modelcheck_graph.png",
    mime="image/png"
)

if not errors_df.empty:
    st.download_button(
        "Download Algorithm Errors CSV",
        errors_df.to_csv(index=False).encode("utf-8"),
        file_name="modelcheck_errors.csv",
        mime="text/csv"
    )

st.caption(
    "Normal evaluates untouched test data. Noise adds random variation to "
    "numeric features. Feature-Shift moves one selected numeric feature. "
    "Automatic task detection is a heuristic; select the task manually when needed."
)
