
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ModelCheck AI",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 ModelCheck AI")
st.subheader("Same Data, Different Models — Robustness Under Feature Perturbations")
st.write(
    "Compare four machine-learning algorithms under normal test data, "
    "random numeric noise, and a selected feature shift."
)

st.info(
    "This app measures performance on a held-out test set under three "
    "conditions. It does not force models to have identical validation scores."
)


# ---------------------------------------------------------
# 1. DATA UPLOAD AND CLEANING
# ---------------------------------------------------------

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

df = raw_df.copy()

# Remove completely empty rows and columns.
df = df.dropna(axis=0, how="all")
df = df.dropna(axis=1, how="all")

# Remove duplicate rows.
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# Remove columns with no usable values.
df = df.dropna(axis=1, how="all")

if df.empty or df.shape[1] < 2:
    st.error(
        "The dataset needs at least two usable columns: "
        "one or more features and one target."
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows after cleaning", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Duplicate rows removed", f"{duplicate_count:,}")

with st.expander("Preview cleaned dataset", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Dataset information"):
    info_df = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Data type": [str(dtype) for dtype in df.dtypes],
            "Missing values": [int(df[col].isna().sum()) for col in df.columns],
            "Unique values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )
    st.dataframe(info_df, use_container_width=True)


# ---------------------------------------------------------
# 2. TARGET SELECTION AND TASK DETECTION
# ---------------------------------------------------------

st.header("2. Select Target")

column_names = list(df.columns)

preferred_targets = [
    col for col in column_names
    if str(col).strip().lower() in {
        "price_usd",
        "performance_score",
        "target",
        "label",
        "class",
    }
]

default_target = (
    preferred_targets[0]
    if preferred_targets
    else column_names[-1]
)

target_col = st.selectbox(
    "Choose the target column you want to predict",
    options=column_names,
    index=column_names.index(default_target),
)

work_df = df.copy()

# Convert numeric-looking targets into numeric values where appropriate.
target_as_numeric = pd.to_numeric(work_df[target_col], errors="coerce")
non_missing_target_count = int(work_df[target_col].notna().sum())

if (
    non_missing_target_count > 0
    and int(target_as_numeric.notna().sum()) == non_missing_target_count
):
    work_df[target_col] = target_as_numeric

# Rows without a target cannot be used for supervised learning.
missing_target_rows = int(work_df[target_col].isna().sum())
if missing_target_rows:
    work_df = work_df.dropna(subset=[target_col]).reset_index(drop=True)

if len(work_df) < 5:
    st.error(
        "There are too few usable rows after removing missing target values. "
        "Please upload a larger dataset."
    )
    st.stop()

y = work_df[target_col].copy()
X = work_df.drop(columns=[target_col]).copy()

# Drop likely identifier or source-file columns.
excluded_columns = []
for col in X.columns:
    normalized = str(col).strip().lower().replace(" ", "_")
    if normalized in {
        "id",
        "name",
        "source_file",
        "filename",
        "file_name",
    }:
        excluded_columns.append(col)

if excluded_columns:
    X = X.drop(columns=excluded_columns)
    st.caption(
        "Excluded likely identifier columns: "
        + ", ".join(map(str, excluded_columns))
    )

if X.shape[1] == 0:
    st.error("No usable feature columns remain after excluding identifiers.")
    st.stop()

# Automatically identify classification versus regression.
if (
    pd.api.types.is_object_dtype(y)
    or isinstance(y.dtype, pd.CategoricalDtype)
    or pd.api.types.is_bool_dtype(y)
):
    task_type = "Classification"
else:
    unique_target_values = int(y.nunique(dropna=True))
    if unique_target_values <= 10:
        task_type = "Classification"
    else:
        task_type = "Regression"


# Fully automatic task detection

if (
    pd.api.types.is_object_dtype(y)
    or isinstance(y.dtype, pd.CategoricalDtype)
    or pd.api.types.is_bool_dtype(y)
):
    task_type = "Classification"

else:
    unique_count = y.nunique(dropna=True)
    total_count = len(y)

    # Binary numeric labels: 0/1
    unique_values = set(y.dropna().unique())

    if unique_count <= 2 and unique_values.issubset({0, 1}):
        task_type = "Classification"

    elif (
        pd.api.types.is_integer_dtype(y)
        and unique_count <= 10
        and unique_count / max(total_count, 1) < 0.05
    ):
        task_type = "Classification"

    else:
        task_type = "Regression"

st.success(f"Detected task: {task_type}")

if task_type == "Classification":
    y = y.astype(str)
if task_type == "Classification" and y.nunique() < 2:
    st.error("Classification needs at least two different target classes.")
    st.stop()

if task_type == "Regression":
    y = pd.to_numeric(y, errors="coerce")
    valid_target_mask = y.notna()
    X = X.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask].copy()

if len(X) < 5:
    st.error("Not enough usable rows remain to train and evaluate models.")
    st.stop()


# ---------------------------------------------------------
# 3. TRAIN / TEST SETTINGS
# ---------------------------------------------------------

st.header("3. Experiment Settings")

settings_col1, settings_col2 = st.columns(2)

with settings_col1:
    test_fraction = st.slider(
        "Test-set percentage",
        min_value=20,
        max_value=40,
        value=20,
        step=5,
    ) / 100.0

with settings_col2:
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=42,
        step=1,
    )

if task_type == "Classification":
    class_counts = y.value_counts()
    can_stratify = (
        y.nunique() > 1
        and int(class_counts.min()) >= 2
        and int(round(len(y) * test_fraction)) >= y.nunique()
    )
    stratify_target = y if can_stratify else None
else:
    stratify_target = None

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=int(random_seed),
        stratify=stratify_target,
    )
except ValueError as exc:
    st.error(
        "Could not split this dataset into training and test sets. "
        f"Check the number of rows and target classes. Details: {exc}"
    )
    st.stop()

if task_type == "Classification":
    if y_train.nunique() < 2:
        st.error(
            "The training split contains fewer than two target classes. "
            "Try a larger dataset or a different random seed."
        )
        st.stop()

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [
    col for col in X_train.columns
    if col not in numeric_cols
]

# Treat numeric columns stored as strings as numeric when all non-missing
# values can be converted. This helps CSV files with mixed formatting.
for col in list(categorical_cols):
    converted_train = pd.to_numeric(X_train[col], errors="coerce")
    non_missing = X_train[col].notna()

    if non_missing.any() and converted_train[non_missing].notna().all():
        X_train[col] = converted_train
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        numeric_cols.append(col)
        categorical_cols.remove(col)

# Preserve the same feature order across train and test sets.
feature_order = list(X_train.columns)
X_train = X_train[feature_order].copy()
X_test = X_test[feature_order].copy()

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [
    col for col in X_train.columns
    if col not in numeric_cols
]

if not numeric_cols and not categorical_cols:
    st.error("No usable features were found in the dataset.")
    st.stop()

st.write(f"Training rows: **{len(X_train)}**")
st.write(f"Testing rows: **{len(X_test)}**")
st.write(f"Numeric features: **{len(numeric_cols)}**")
st.write(f"Categorical features: **{len(categorical_cols)}**")


# ---------------------------------------------------------
# 4. NOISE AND FEATURE-SHIFT SETTINGS
# ---------------------------------------------------------

st.header("4. Noise and Feature-Shift Experiments")

noise_col, shift_col = st.columns(2)

with noise_col:
    noise_strength = st.slider(
        "Noise strength (standard deviations)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Random Gaussian noise added to numeric test features.",
    )

with shift_col:
    feature_shift_strength = st.slider(
        "Feature-Shift strength (standard deviations)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.25,
        help="Shift one numeric test feature by this many training standard deviations.",
    )

if numeric_cols:
    default_feature = numeric_cols[0]
    feature_to_shift = st.selectbox(
        "Numeric feature to shift",
        options=numeric_cols,
        index=numeric_cols.index(default_feature),
    )
else:
    feature_to_shift = None
    st.warning(
        "No numeric feature is available for numeric Noise or Feature-Shift. "
        "Those conditions will match the normal test features."
    )


# ---------------------------------------------------------
# 5. CREATE PERTURBED TEST SETS

# ---------------------------------------------------------
# 5. CREATE PERTURBED TEST SETS
# ---------------------------------------------------------

X_normal = X_test.copy()

# Noise condition: safely convert numeric columns to float.
X_noise = X_test.copy()
noise_rng = np.random.default_rng(int(random_seed))

for col in numeric_cols:
    values = pd.to_numeric(
        X_noise[col], errors="coerce"
    ).astype("float64").to_numpy(copy=True)

    train_values = pd.to_numeric(
        X_train[col], errors="coerce"
    ).astype("float64").to_numpy()

    finite_train = train_values[np.isfinite(train_values)]

    scale = (
        float(np.std(finite_train, ddof=1))
        if finite_train.size > 1
        else 0.0
    )

    if not np.isfinite(scale) or scale < 0:
        scale = 0.0

    noise = noise_rng.normal(
        loc=0.0,
        scale=float(noise_strength) * scale,
        size=len(values),
    )

    valid_values = np.isfinite(values)
    values[valid_values] += noise[valid_values]

    X_noise[col] = pd.Series(
        values,
        index=X_noise.index,
        dtype="float64",
    )

# Feature-Shift condition.
X_shift = X_test.copy()

if feature_to_shift is not None:
    shift_values = pd.to_numeric(
        X_shift[feature_to_shift], errors="coerce"
    ).astype("float64").to_numpy(copy=True)

    train_shift = pd.to_numeric(
        X_train[feature_to_shift], errors="coerce"
    ).astype("float64").to_numpy()

    finite_shift = train_shift[np.isfinite(train_shift)]

    shift_scale = (
        float(np.std(finite_shift, ddof=1))
        if finite_shift.size > 1
        else 0.0
    )

    if not np.isfinite(shift_scale) or shift_scale < 0:
        shift_scale = 0.0

    valid_shift = np.isfinite(shift_values)
    shift_values[valid_shift] += (
        float(feature_shift_strength) * shift_scale
    )

    X_shift[feature_to_shift] = pd.Series(
        shift_values,
        index=X_shift.index,
        dtype="float64",
    )

# Keep feature columns in the same order.
X_normal = X_normal[feature_order].copy()
X_noise = X_noise[feature_order].copy()
X_shift = X_shift[feature_order].copy()

test_conditions = {
    "Normal": X_normal,
    "Noise": X_noise,
    "Feature-Shift": X_shift,
}



# ---------------------------------------------------------
# 6. DEFINE FOUR ALGORITHMS
# ---------------------------------------------------------

if task_type == "Classification":
    algorithms = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=int(random_seed),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=int(random_seed),
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=int(random_seed),
        ),
        "SVM": SVC(
            random_state=int(random_seed),
        ),
    }
    primary_metric = "Accuracy"
else:
    algorithms = {
        "Decision Tree": DecisionTreeRegressor(
            random_state=int(random_seed),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=int(random_seed),
            n_jobs=-1,
        ),
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
    }
    primary_metric = "RMSE"


# ---------------------------------------------------------
# 7. PREPROCESSING AND METRICS
# ---------------------------------------------------------

def build_pipeline(model, numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def calculate_metrics(y_true, y_pred, current_task):
    if current_task == "Classification":
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Balanced Accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "Precision (weighted)": float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "Recall (weighted)": float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "F1 Score (weighted)": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

    y_true_numeric = np.asarray(y_true, dtype=float)
    y_pred_numeric = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_numeric, y_pred_numeric)))

    try:
        r2 = float(r2_score(y_true_numeric, y_pred_numeric))
    except Exception:
        r2 = np.nan

    return {
        "MAE": float(mean_absolute_error(y_true_numeric, y_pred_numeric)),
        "RMSE": rmse,
        "R2 Score": r2,
    }


# ---------------------------------------------------------
# 8. TRAIN AND COMPARE MODELS
# ---------------------------------------------------------

st.header("5. Train and Compare Models")

run_models = st.button(
    "Train and Compare Models",
    type="primary",
    use_container_width=True,
)

if run_models:
    all_metrics = []
    all_predictions = []
    normal_prediction_rows = []
    fitted_models = {}

    progress = st.progress(0)
    status = st.empty()

    total_steps = len(algorithms) * (1 + len(test_conditions))
    current_step = 0

    test_index = X_test.index

    try:
        for model_name, model in algorithms.items():
            status.write(f"Training {model_name}...")

            pipeline = build_pipeline(
                model,
                numeric_cols,
                categorical_cols,
            )

            pipeline.fit(X_train, y_train)
            fitted_models[model_name] = pipeline

            for condition_name, X_condition in test_conditions.items():
                status.write(
                    f"Evaluating {model_name} on {condition_name} data..."
                )

                predictions = pipeline.predict(X_condition)
                metrics = calculate_metrics(
                    y_test,
                    predictions,
                    task_type,
                )

                metric_row = {
                    "Algorithm": model_name,
                    "Condition": condition_name,
                    **metrics,
                }
                all_metrics.append(metric_row)

                condition_prediction_df = pd.DataFrame(
                    {
                        "Test Row Index": test_index.to_numpy(),
                        "Algorithm": model_name,
                        "Condition": condition_name,
                        "Actual": np.asarray(y_test),
                        "Predicted": np.asarray(predictions),
                    }
                )
                all_predictions.append(condition_prediction_df)

                if condition_name == "Normal":
                    normal_prediction_rows.append(
                        pd.DataFrame(
                            {
                                "Test Row Index": test_index.to_numpy(),
                                "Actual": np.asarray(y_test),
                                f"{model_name} Prediction": np.asarray(
                                    predictions
                                ),
                            }
                        )
                    )

                current_step += 1
                progress.progress(min(current_step / total_steps, 1.0))

        metrics_df = pd.DataFrame(all_metrics)
        predictions_df = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        # Combine normal predictions into one comparison table.
        normal_predictions_df = normal_prediction_rows[0]
        for next_df in normal_prediction_rows[1:]:
            normal_predictions_df = normal_predictions_df.merge(
                next_df,
                on=["Test Row Index", "Actual"],
                how="outer",
            )

        # Build a model-by-condition comparison table.
        if task_type == "Classification":
            comparison_metric = "Accuracy"
            metric_direction = "higher"
        else:
            comparison_metric = "RMSE"
            metric_direction = "lower"

        comparison_df = metrics_df[
            ["Algorithm", "Condition", comparison_metric]
        ].copy()

        comparison_wide = comparison_df.pivot(
            index="Algorithm",
            columns="Condition",
            values=comparison_metric,
        ).reset_index()

        # Calculate change from Normal for each model.
        delta_rows = []

        for model_name in algorithms:
            model_results = metrics_df[
                metrics_df["Algorithm"] == model_name
            ].set_index("Condition")

            normal_value = model_results.loc[
                "Normal", comparison_metric
            ]

            for condition_name in ["Noise", "Feature-Shift"]:
                perturbed_value = model_results.loc[
                    condition_name, comparison_metric
                ]

                if pd.notna(normal_value) and pd.notna(perturbed_value):
                    change = float(perturbed_value - normal_value)
                else:
                    change = np.nan

                if metric_direction == "higher":
                    interpretation = (
                        "Improved" if change > 0
                        else "Declined" if change < 0
                        else "No change"
                    )
                else:
                    interpretation = (
                        "Improved" if change < 0
                        else "Declined" if change > 0
                        else "No change"
                    )

                delta_rows.append(
                    {
                        "Algorithm": model_name,
                        "Condition": condition_name,
                        f"Normal {comparison_metric}": normal_value,
                        f"Perturbed {comparison_metric}": perturbed_value,
                        "Change vs Normal": change,
                        "Interpretation": interpretation,
                    }
                )

        delta_df = pd.DataFrame(delta_rows)

        # Store all outputs in session state for downloads and reruns.
        st.session_state["modelcheck_metrics"] = metrics_df
        st.session_state["modelcheck_predictions"] = predictions_df
        st.session_state["modelcheck_normal_predictions"] = normal_predictions_df
        st.session_state["modelcheck_comparison"] = comparison_wide
        st.session_state["modelcheck_delta"] = delta_df
        st.session_state["modelcheck_metric_name"] = comparison_metric
        st.session_state["modelcheck_task"] = task_type

        status.success("All four algorithms have been trained and evaluated.")
        progress.progress(1.0)

    except Exception as exc:
        status.empty()
        progress.empty()
        st.error(
            "Model training or evaluation failed. "
            "Check the dataset and target column, then try again."
        )
        st.exception(exc)


# ---------------------------------------------------------
# 9. DISPLAY SAVED RESULTS
# ---------------------------------------------------------

if "modelcheck_metrics" in st.session_state:
    metrics_df = st.session_state["modelcheck_metrics"]
    predictions_df = st.session_state["modelcheck_predictions"]
    normal_predictions_df = st.session_state["modelcheck_normal_predictions"]
    comparison_wide = st.session_state["modelcheck_comparison"]
    delta_df = st.session_state["modelcheck_delta"]
    comparison_metric = st.session_state["modelcheck_metric_name"]
    result_task = st.session_state["modelcheck_task"]

    st.header("6. Model Comparison Results")

    st.subheader("All Metrics by Algorithm and Condition")
    st.dataframe(
        metrics_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"{comparison_metric}: Normal vs Perturbed Data")

    chart_data = metrics_df.pivot(
        index="Algorithm",
        columns="Condition",
        values=comparison_metric,
    )

    # Put conditions in a consistent display order.
    desired_order = [
        condition for condition in ["Normal", "Noise", "Feature-Shift"]
        if condition in chart_data.columns
    ]
    chart_data = chart_data[desired_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    chart_data.plot(kind="bar", ax=ax)
    ax.set_title(f"{comparison_metric} by Algorithm and Test Condition")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(comparison_metric)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    graph_buffer = io.BytesIO()
    fig.savefig(
        graph_buffer,
        format="png",
        dpi=200,
        bbox_inches="tight",
    )
    graph_buffer.seek(0)
    plt.close(fig)

    st.subheader("Condition Comparison Table")
    st.dataframe(
        comparison_wide.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Change Compared with Normal Test Data")
    st.dataframe(
        delta_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "For Accuracy, a positive change means performance increased. "
        "For RMSE, a negative change means error decreased. "
        "These are test-set results, not validation scores."
    )

    st.header("7. Predictions")

    st.subheader("Normal Test-Set Predictions")
    st.dataframe(
        normal_predictions_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Predictions for All Conditions")
    st.dataframe(
        predictions_df.head(200),
        use_container_width=True,
        hide_index=True,
    )

    st.header("8. Download Results")

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            "Download All Metrics CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="modelcheck_all_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download Normal Predictions CSV",
            data=normal_predictions_df.to_csv(index=False).encode("utf-8"),
            file_name="modelcheck_normal_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download Condition Comparison CSV",
            data=comparison_wide.to_csv(index=False).encode("utf-8"),
            file_name="modelcheck_condition_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col2:
        st.download_button(
            "Download All Predictions CSV",
            data=predictions_df.to_csv(index=False).encode("utf-8"),
            file_name="modelcheck_all_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download Reliability Change CSV",
            data=delta_df.to_csv(index=False).encode("utf-8"),
            file_name="modelcheck_reliability_changes.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download Comparison Graph PNG",
            data=graph_buffer.getvalue(),
            file_name="modelcheck_comparison_graph.png",
            mime="image/png",
            use_container_width=True,
        )

    st.divider()
    st.subheader("How to Interpret the Results")
    st.markdown(
        """
        - **Normal:** Performance on the original held-out test data.
        - **Noise:** Performance after random Gaussian noise is added to
          numeric test features.
        - **Feature-Shift:** Performance after shifting the selected numeric
          feature by a chosen number of training-set standard deviations.
        - **Robustness:** A smaller performance deterioration under a
          perturbation suggests greater robustness in this experiment.
        - **Important limitation:** One random noise realization is used per
          run. For stronger research evidence, repeat the experiment with
          multiple seeds and report average performance and variation.
        """
    )

else:
    st.info(
        "Set your experiment options above, then click "
        "**Train and Compare Models** to generate results and downloads."
    )
