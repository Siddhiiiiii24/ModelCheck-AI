
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR


# ======================== PAGE CONFIGURATION ========================

st.set_page_config(
    page_title="ModelCheck AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ModelCheck AI")
st.write(
    "Upload a dataset, train multiple algorithms and compare "
    "their performance under Normal, Noise and Feature-Shift conditions."
)

uploaded = st.file_uploader(
    "Upload ONE dataset CSV at a time",
    type=["csv"]
)

if uploaded is None:
    st.info("Upload car_dataset.csv OR toy_dataset.csv to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.empty or len(df) < 5:
    st.error("Dataset must contain at least 5 rows.")
    st.stop()

st.subheader("1. Dataset")
st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
st.dataframe(df.head(10), use_container_width=True)

# Preserve original cleaning functionality
df = df.dropna(axis=1, how="all").drop_duplicates()

if len(df) < 5 or len(df.columns) < 2:
    st.error("Dataset needs at least 5 rows and a target plus one feature.")
    st.stop()


# ======================== TARGET SELECTION ========================

st.subheader("2. Configure Experiment")

default_target = (
    "Price_USD" if "Price_USD" in df.columns
    else "Performance_Score" if "Performance_Score" in df.columns
    else df.columns[-1]
)

target = st.selectbox(
    "Choose the target column",
    df.columns.tolist(),
    index=df.columns.tolist().index(default_target)
)

# Convert numeric-looking target values such as ₹50,000
target_text = (
    df[target]
    .astype(str)
    .str.strip()
    .str.replace(r"[$₹,]", "", regex=True)
)

converted_target = pd.to_numeric(target_text, errors="coerce")
non_empty = df[target].notna()

numeric_ratio = (
    converted_target[non_empty].notna().mean()
    if non_empty.any() else 0
)

if numeric_ratio >= 0.80:
    df[target] = converted_target

data = df.dropna(subset=[target]).copy()

X = data.drop(columns=[target])
y = data[target]

# Preserve original excluded-column logic
excluded = [
    c for c in X.columns
    if c.strip().lower() in ["id", "name", "source file"]
]
X = X.drop(columns=excluded, errors="ignore")

if X.shape[1] == 0:
    st.error("No usable input features remain.")
    st.stop()

if y.nunique() < 2:
    st.error("The target must contain at least two different values.")
    st.stop()

numeric_target = pd.api.types.is_numeric_dtype(y)
unique_count = y.nunique()
target_name = target.strip().lower()

regression_names = [
    "salary", "price", "income", "wage",
    "performance_score", "age"
]

if numeric_target and (
    unique_count > 10 or target_name in regression_names
):
    task = "regression"
else:
    task = "classification"

st.success(f"Prediction type: {task.title()}")
st.write(f"Target column: **{target}**")
st.write(f"Unique target values: **{unique_count}**")

if task == "classification" and y.value_counts().min() < 2:
    st.warning(
        "At least one class has fewer than 2 samples. "
        "The train/test split may be unstable."
    )

if len(data) < 10:
    st.warning("This is a small dataset. Scores may be unstable.")


# ======================== FOUR ALGORITHMS ========================

if task == "classification":
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": SVC()
    }
else:
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "SVR": SVR()
    }

selected = st.multiselect(
    "Choose algorithms to train and compare",
    options=list(models.keys()),
    default=list(models.keys()),
    key="algorithm_selection"
)

test_size = st.slider(
    "Test data percentage",
    min_value=20,
    max_value=40,
    value=20,
    step=5
)


# ======================== NOISE AND FEATURE SHIFT ========================

st.subheader("3. Robustness Testing")

st.write(
    "Compare models on the same test rows under Normal, Noise "
    "and Feature-Shift conditions."
)

numeric_features = X.select_dtypes(include="number").columns.tolist()

col1, col2 = st.columns(2)

with col1:
    noise_strength = st.slider(
        "Noise strength (standard deviations)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

with col2:
    shift_strength = st.slider(
        "Feature-Shift strength (standard deviations)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.5
    )

if numeric_features:
    shift_feature = st.selectbox(
        "Select feature for Feature-Shift",
        numeric_features
    )
else:
    shift_feature = None
    st.warning(
        "No numeric input features found. Numeric perturbations "
        "cannot be applied to this dataset."
    )

run = st.button(
    "🚀 Train and Compare Models",
    type="primary"
)


# ======================== PREPROCESSING FUNCTION ========================

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipe, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipe, categorical_cols)
        )

    return ColumnTransformer(transformers=transformers)


# ======================== METRICS FUNCTION ========================

def calculate_metrics(y_true, y_pred, task_type):

    if task_type == "classification":
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "Precision (weighted)": precision_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0
            ),
            "Recall (weighted)": recall_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0
            ),
            "F1 Score (weighted)": f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0
            )
        }

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2 Score": r2_score(y_true, y_pred)
    }


def csv_bytes(frame, index=False):
    return frame.to_csv(index=index).encode("utf-8-sig")


# ======================== RUN EXPERIMENT ========================

if run:

    if not selected:
        st.error("Please select at least one algorithm.")
        st.stop()

    stratify = None

    if task == "classification":
        counts = y.value_counts()

        if len(counts) >= 2 and counts.min() >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42,
            stratify=stratify
        )
    except Exception as e:
        st.error(f"Could not split dataset: {e}")
        st.stop()

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Use training-set statistics only
    train_std = (
        X_train[numeric_cols].std(numeric_only=True)
        if numeric_cols else pd.Series(dtype=float)
    )

    # Three versions of the same test dataset
    X_normal = X_test.copy()
    X_noise = X_test.copy()
    X_shift = X_test.copy()

    # Add reproducible Gaussian noise to numeric input features
    rng = np.random.default_rng(42)

    for col in numeric_cols:
        scale = train_std.get(col, 0)

        if pd.notna(scale) and scale > 0 and noise_strength > 0:
            valid = X_noise[col].notna()
            noise = rng.normal(
                0,
                noise_strength * scale,
                size=len(X_noise)
            )

            X_noise.loc[valid, col] = (
                X_noise.loc[valid, col].astype(float)
                + noise[valid.to_numpy()]
            )

    # Shift one selected numeric feature
    if shift_feature is not None and shift_strength > 0:
        scale = train_std.get(shift_feature, 0)

        if pd.notna(scale) and scale > 0:
            valid = X_shift[shift_feature].notna()

            X_shift.loc[valid, shift_feature] = (
                X_shift.loc[valid, shift_feature].astype(float)
                + shift_strength * scale
            )

    conditions = {
        "Normal": X_normal,
        "Noise": X_noise,
        "Feature-Shift": X_shift
    }

    all_results = []
    all_predictions = []
    normal_predictions = pd.DataFrame(index=y_test.index)

    progress = st.progress(0)
    status = st.empty()

    for i, name in enumerate(selected):

        status.write(f"Training {name}...")

        try:
            pipeline = Pipeline([
                (
                    "preprocessing",
                    build_preprocessor(numeric_cols, categorical_cols)
                ),
                ("model", models[name])
            ])

            # Fit only on training data
            pipeline.fit(X_train, y_train)

            # Evaluate same fitted model under all conditions
            for condition_name, X_condition in conditions.items():

                pred = pipeline.predict(X_condition)

                metrics = calculate_metrics(
                    y_test, pred, task
                )

                all_results.append({
                    "Algorithm": name,
                    "Condition": condition_name,
                    **metrics
                })

                all_predictions.append(pd.DataFrame({
                    "Test_Row_Index": y_test.index.to_numpy(),
                    "Actual": y_test.to_numpy(),
                    "Algorithm": name,
                    "Condition": condition_name,
                    "Prediction": pred
                }))

                if condition_name == "Normal":
                    normal_predictions[name] = pred

        except Exception as e:
            st.warning(f"{name} failed: {e}")

        progress.progress((i + 1) / len(selected))

    status.empty()

    if not all_results:
        st.error("No models trained successfully.")
        st.stop()

    result_df = pd.DataFrame(all_results)
    detailed_predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    # ======================== RESULTS TABLE ========================

    st.subheader("4. Model Comparison — All Conditions")
    st.dataframe(result_df, use_container_width=True)

    # ======================== COMPARISON GRAPH ========================

    st.subheader("5. Performance Comparison Graph")

    if task == "classification":
        chart_metric = "Accuracy"
        plot_multiplier = 100
        ylabel = "Accuracy (%)"
    else:
        chart_metric = "RMSE"
        plot_multiplier = 1
        ylabel = "RMSE (lower is better)"

    plot_data = result_df.pivot(
        index="Algorithm",
        columns="Condition",
        values=chart_metric
    )

    plot_data = plot_data * plot_multiplier

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data.plot(kind="bar", ax=ax)

    ax.set_title(
        f"{chart_metric}: Normal vs Noise vs Feature-Shift"
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Condition")

    fig.tight_layout()
    st.pyplot(fig)

    # Store PNG graph in memory for download
    graph_buffer = io.BytesIO()
    fig.savefig(
        graph_buffer,
        format="png",
        dpi=200,
        bbox_inches="tight"
    )
    graph_buffer.seek(0)
    plt.close(fig)

    # ======================== CHANGE ANALYSIS ========================

    st.subheader("6. Normal vs Noise vs Feature-Shift Analysis")

    metric_columns = [
        c for c in result_df.columns
        if c not in ["Algorithm", "Condition"]
    ]

    normal_metrics = (
        result_df[result_df["Condition"] == "Normal"]
        .drop(columns="Condition")
        .set_index("Algorithm")
    )

    change_rows = []

    for condition_name in ["Noise", "Feature-Shift"]:

        condition_metrics = (
            result_df[result_df["Condition"] == condition_name]
            .drop(columns="Condition")
            .set_index("Algorithm")
        )

        for algorithm in condition_metrics.index:

            for metric in metric_columns:

                normal_value = normal_metrics.loc[
                    algorithm, metric
                ]

                condition_value = condition_metrics.loc[
                    algorithm, metric
                ]

                change_rows.append({
                    "Algorithm": algorithm,
                    "Condition": condition_name,
                    "Metric": metric,
                    "Normal Score": normal_value,
                    "Condition Score": condition_value,
                    "Change (Condition - Normal)": (
                        condition_value - normal_value
                    )
                })

    change_df = pd.DataFrame(change_rows)

    st.dataframe(change_df, use_container_width=True)

    st.caption(
        "For Accuracy, F1, Precision, Recall and R², a negative change "
        "usually indicates worse performance. For MAE and RMSE, a "
        "positive change usually indicates worse performance."
    )

    # ======================== INDIVIDUAL PREDICTIONS ========================

    st.subheader("7. Individual Predictions")

    normal_comparison = pd.DataFrame({
        "Actual": y_test
    })

    for name in normal_predictions.columns:
        normal_comparison[f"{name} Prediction"] = normal_predictions[name]

    st.write("Normal test condition:")
    st.dataframe(normal_comparison, use_container_width=True)

    st.write("Detailed predictions for every algorithm and condition:")
    st.dataframe(detailed_predictions, use_container_width=True)

    # ======================== DOWNLOAD OPTIONS ========================

    st.subheader("8. Download Results")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Download Metrics CSV",
            data=csv_bytes(result_df),
            file_name="modelcheck_metrics_all_conditions.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Download Normal Predictions CSV",
            data=csv_bytes(normal_comparison, index=True),
            file_name="modelcheck_normal_predictions.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Download Comparison CSV",
            data=csv_bytes(change_df),
            file_name="modelcheck_condition_comparison.csv",
            mime="text/csv"
        )

    with d2:
        st.download_button(
            "⬇️ Download All Predictions CSV",
            data=csv_bytes(detailed_predictions),
            file_name="modelcheck_predictions_all_conditions.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Download Comparison Graph PNG",
            data=graph_buffer.getvalue(),
            file_name="modelcheck_comparison_graph.png",
            mime="image/png"
        )

    # ======================== INTERPRETATION ========================

    st.subheader("9. Experiment Notes")

    st.write(
        "**Normal:** The original test features are used. "
        "**Noise:** Random Gaussian noise is added to numeric input features. "
        "**Feature-Shift:** One selected numeric feature is shifted by a "
        "multiple of its standard deviation calculated from training data."
    )

    st.caption(
        "All conditions use the same held-out test rows and the same fitted "
        "models. The target values are not changed. These are test performance "
        "results, not validation scores. One reproducible noise realization is "
        "used; repeated runs with multiple seeds are recommended for stronger "
        "research conclusions."
    )
'''

print("Complete code provided above.") '''
