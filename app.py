
import io
import pandas as pd
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

st.set_page_config(
    page_title="ModelCheck AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ModelCheck AI")
st.write("Upload a dataset, select algorithms, train models and compare their performance.")

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

# Remove completely empty columns and duplicate rows
df = df.dropna(axis=1, how="all").drop_duplicates()

# Select target column
st.subheader("2. Configure experiment")

default_target = (
    "Price_USD" if "Price_USD" in df.columns
    else "Performance_Score" if "Performance_Score" in df.columns
    else df.columns[-1]
)

target = st.selectbox(
    "Choose the target column (what should the model predict?)",
    df.columns.tolist(),
    index=df.columns.tolist().index(default_target)
)

# Remove rows where target is missing
data = df.dropna(subset=[target]).copy()

# Exclude IDs and names from predictors by default
X = data.drop(columns=[target])
y = data[target]

excluded = [
    c for c in X.columns
    if c.strip().lower() in ["id", "name", "source file"]
]
X = X.drop(columns=excluded, errors="ignore")

if X.shape[1] == 0:
    st.error("No usable input features remain. Select a different target column.")
    st.stop()

# Infer task from target data
numeric_target = pd.api.types.is_numeric_dtype(y)
unique_count = y.nunique()

if numeric_target and unique_count > 10:
    task = "regression"
else:
    task = "classification"

st.info(
    f"Detected task: **{task.title()}**. "
    f"Target: **{target}**. "
    "Check that this matches your intended prediction task."
)

if task == "classification" and unique_count < 2:
    st.error("Classification needs at least two target classes.")
    st.stop()

if len(data) < 10:
    st.warning("This dataset is very small. Scores may be unstable.")

if task == "classification" and y.value_counts().min() < 2:
    st.warning(
        "At least one class has fewer than 2 rows. "
        "A reliable stratified split may not be possible."
    )

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

run = st.button("🚀 Train and Compare Models", type="primary")

if run:
    if not selected:
        st.error("Select at least one algorithm.")
        st.stop()

    stratify = None
    if task == "classification":
        counts = y.value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size / 100,
            random_state=42,
            stratify=stratify
        )
    except Exception as e:
        st.error(f"Could not split dataset: {e}")
        st.stop()

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

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
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    results = []
    predictions = pd.DataFrame(index=y_test.index)
    fitted_models = {}

    progress = st.progress(0)
    status = st.empty()

    for i, name in enumerate(selected):
        status.write(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", models[name])
        ])

        try:
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            fitted_models[name] = pipeline
            predictions[name] = pred

            if task == "classification":
                results.append({
                    "Algorithm": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
                    "Precision (weighted)": precision_score(
                        y_test, pred, average="weighted", zero_division=0
                    ),
                    "Recall (weighted)": recall_score(
                        y_test, pred, average="weighted", zero_division=0
                    ),
                    "F1 Score (weighted)": f1_score(
                        y_test, pred, average="weighted", zero_division=0
                    )
                })
            else:
                results.append({
                    "Algorithm": name,
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                    "R2 Score": r2_score(y_test, pred)
                })

        except Exception as e:
            st.warning(f"{name} failed: {e}")

        progress.progress((i + 1) / len(selected))

    status.empty()

    if not results:
        st.error("No models trained successfully. Check your target and dataset.")
        st.stop()

    result_df = pd.DataFrame(results)

    st.subheader("3. Model comparison")
    st.dataframe(result_df, use_container_width=True)

    if task == "classification":
        chart_metric = "Accuracy"
        chart_data = result_df.set_index("Algorithm")[chart_metric] * 100
        ylabel = "Accuracy (%)"
    else:
        chart_metric = "RMSE"
        chart_data = result_df.set_index("Algorithm")[chart_metric]
        ylabel = "RMSE (lower is better)"

    fig, ax = plt.subplots(figsize=(9, 4))
    chart_data.plot(kind="bar", ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. Individual predictions")
    comparison = pd.DataFrame({
        "Actual": y_test,
    })
    for name in predictions.columns:
        comparison[f"{name} Prediction"] = predictions[name]

    st.dataframe(comparison, use_container_width=True)

    st.subheader("5. Download results")
    st.download_button(
        "⬇️ Download metrics CSV",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="modelcheck_metrics.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download sample predictions CSV",
        data=comparison.to_csv(index=True).encode("utf-8-sig"),
        file_name="modelcheck_predictions.csv",
        mime="text/csv"
    )

    st.caption(
        "Models are trained on the uploaded dataset using a fixed random split. "
        "Results depend on the dataset, selected target and split."
    )
