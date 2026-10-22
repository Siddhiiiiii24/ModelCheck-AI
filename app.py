import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ModelCheck AI | Model Reliability Analyzer",
    page_icon="📊",
    layout="wide",
)

# ----------------------------- Styling ---------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
.hero {
  padding: 1.4rem 1.6rem; border-radius: 18px;
  background: linear-gradient(120deg, #102a56 0%, #2456a6 100%);
  color: white; margin-bottom: 1.2rem;
}
.hero h1 {color: white; margin: 0; font-size: 2.15rem;}
.hero p {color: #e6efff; margin: .35rem 0 0 0; font-size: 1.02rem;}
.small-note {font-size: .9rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📊 ModelCheck AI</h1>
  <p>Model Reliability Analyzer · Compare validation performance and behavior under data shifts</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Upload your experiment CSV files to compare machine-learning model configurations, "
    "inspect performance under different test conditions, and export a combined analysis."
)

with st.sidebar:
    st.header("About this project")
    st.write(
        "This dashboard supports the research demonstration "
        "**“Underspecification: Same Validation Score, Different Deployment Behavior.”**"
    )
    st.divider()
    st.subheader("How to use")
    st.markdown(
        "1. Upload one or more CSV result files.\\n"
        "2. Check the detected columns.\\n"
        "3. Compare model performance and reliability.\\n"
        "4. Download the combined data and summary."
    )
    st.caption("This tool analyzes recorded metrics; it does not retrain models.")

# ----------------------------- Data loading ----------------------------------
uploaded_files = st.file_uploader(
    "Upload experiment result CSV files",
    type=["csv"],
    accept_multiple_files=True,
    help="Upload the CSV results exported by your Logistic Regression/SVM and Decision Tree/Random Forest experiments."
)

def read_csv_upload(upload):
    try:
        data = pd.read_csv(upload)
        if data.empty:
            st.warning(f"{upload.name} is empty and was skipped.")
            return None
        data["Source File"] = upload.name
        return data
    except Exception as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return None

def find_column(df, candidates):
    # Case-insensitive, whitespace-insensitive matching
    normalized = {str(c).strip().lower().replace("_", " ").replace("-", " "): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ").replace("-", " ")
        if key in normalized:
            return normalized[key]
    # allow partial matching for common labels
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ").replace("-", " ")
        for norm, original in normalized.items():
            if key in norm or norm in key:
                return original
    return None

def normalize_results(df):
    """Convert the two project CSV formats and similar tidy CSVs into a common schema."""
    data = df.copy()
    algorithm_col = find_column(data, ["Algorithm", "Model", "Classifier"])
    config_col = find_column(data, ["Configuration", "Config", "Parameters", "Model Configuration"])
    env_col = find_column(data, ["Environment", "Test Condition", "Condition", "Dataset Type", "Evaluation Set"])
    val_col = find_column(data, ["Validation Accuracy", "Validation Score", "Val Accuracy", "Validation"])
    acc_col = find_column(data, ["Accuracy", "Test Accuracy", "Score"])
    bal_col = find_column(data, ["Balanced Accuracy", "Balanced Acc", "Test Balanced Accuracy"])

    # Support wide-format tables with columns such as Normal Test Accuracy, Noise Test Accuracy.
    if env_col is None and acc_col is None:
        possible = []
        for col in data.columns:
            name = str(col).lower()
            if any(term in name for term in ["normal", "noise", "shift"]) and "accuracy" in name:
                possible.append(col)
        if possible:
            id_vars = [c for c in data.columns if c not in possible]
            data = data.melt(id_vars=id_vars, value_vars=possible, var_name="Environment", value_name="Accuracy")
            env_col = "Environment"
            acc_col = "Accuracy"
            algorithm_col = find_column(data, ["Algorithm", "Model", "Classifier"])
            config_col = find_column(data, ["Configuration", "Config", "Parameters", "Model Configuration"])
            val_col = find_column(data, ["Validation Accuracy", "Validation Score", "Val Accuracy", "Validation"])
            bal_col = find_column(data, ["Balanced Accuracy", "Balanced Acc", "Test Balanced Accuracy"])

    out = pd.DataFrame(index=data.index)
    out["Algorithm"] = data[algorithm_col].astype(str) if algorithm_col else "Unknown model"
    out["Configuration"] = data[config_col].astype(str) if config_col else "Default / not specified"
    out["Environment"] = data[env_col].astype(str) if env_col else "Unspecified condition"

    if val_col:
        out["Validation Accuracy"] = pd.to_numeric(data[val_col], errors="coerce")
    else:
        out["Validation Accuracy"] = pd.NA

    if acc_col:
        out["Accuracy"] = pd.to_numeric(data[acc_col], errors="coerce")
    else:
        out["Accuracy"] = pd.NA

    if bal_col:
        out["Balanced Accuracy"] = pd.to_numeric(data[bal_col], errors="coerce")
    else:
        out["Balanced Accuracy"] = pd.NA

    source_col = find_column(data, ["Source File"])
    out["Source File"] = data[source_col].astype(str) if source_col else "Uploaded CSV"

    # Standardize common environment labels without changing the original source data.
    def clean_env(value):
        v = str(value).strip().lower()
        if any(t in v for t in ["noise", "gaussian"]):
            return "Noise Test"
        if any(t in v for t in ["shift", "distribution"]):
            return "Feature-Shift Test"
        if any(t in v for t in ["normal", "id test", "in-distribution", "in distribution", "original"]):
            return "Normal Test"
        return str(value).strip()
    out["Environment"] = out["Environment"].map(clean_env)
    return out

if not uploaded_files:
    st.info("Start by uploading your experiment CSV files. The dashboard will populate after a file is uploaded.")
    st.subheader("Expected data")
    st.markdown(
        "- **Algorithm**: Logistic Regression, SVM, Decision Tree, or Random Forest\\n"
        "- **Configuration**: model settings (if recorded)\\n"
        "- **Validation Accuracy**: validation score (if recorded)\\n"
        "- **Environment**: Normal Test, Noise Test, or Feature-Shift Test\\n"
        "- **Accuracy** and/or **Balanced Accuracy**: recorded metric values"
    )
    st.warning(
        "Use your real experiment CSVs for research conclusions. No experimental result is generated by this empty dashboard."
    )
    st.stop()

frames = [read_csv_upload(f) for f in uploaded_files]
frames = [f for f in frames if f is not None]
if not frames:
    st.error("No readable CSV files were found.")
    st.stop()

raw = pd.concat(frames, ignore_index=True, sort=False)
normalized = normalize_results(raw)
normalized = normalized.dropna(how="all", subset=["Validation Accuracy", "Accuracy", "Balanced Accuracy"])

st.subheader("1. Uploaded data")
c1, c2, c3 = st.columns(3)
c1.metric("Files uploaded", len(frames))
c2.metric("Rows loaded", len(raw))
c3.metric("Models detected", normalized["Algorithm"].nunique() if not normalized.empty else 0)

with st.expander("View original uploaded data", expanded=False):
    st.dataframe(raw, use_container_width=True, hide_index=True)
with st.expander("View standardized data used for analysis", expanded=False):
    st.dataframe(normalized, use_container_width=True, hide_index=True)

if normalized.empty:
    st.error("No usable metric values were found. Check that the uploaded CSV contains numeric accuracy or validation-accuracy columns.")
    st.write("Columns detected:", ", ".join(map(str, raw.columns)))
    st.stop()

st.caption("Detected source columns: " + ", ".join(map(str, raw.columns)))

# ----------------------------- Filters ---------------------------------------
# ----------------------------- Filters ---------------------------------------
st.subheader("2. Filter and compare")

algorithms = sorted(
    normalized["Algorithm"].dropna().astype(str).unique().tolist()
)

f1, f2 = st.columns(2)
chosen_algorithms = []

with f1:
    st.markdown("### Select Algorithms")

    for algorithm in algorithms:
        if st.checkbox(
            algorithm,
            value=True,
            key=f"modelcheck_fix2_algorithm_{algorithm}"
        ):
            chosen_algorithms.append(algorithm)

if not chosen_algorithms:
    st.warning("Please select at least one algorithm.")
    st.stop()

# Show only test conditions available for selected algorithms
available_conditions = sorted(
    normalized[
        normalized["Algorithm"].astype(str).isin(chosen_algorithms)
    ]["Environment"].dropna().astype(str).unique().tolist()
)

chosen_conditions = []

with f2:
    st.markdown("### Select Test Conditions")

    for condition in available_conditions:
        if st.checkbox(
            condition,
            value=True,
            key=f"modelcheck_fix2_condition_{condition}"
        ):
            chosen_conditions.append(condition)

if not chosen_conditions:
    st.warning("Please select at least one test condition.")
    st.stop()

# Apply selected filters
view = normalized[
    normalized["Algorithm"].astype(str).isin(chosen_algorithms)
    & normalized["Environment"].astype(str).isin(chosen_conditions)
].copy()

if view.empty:
    st.warning(
        "No matching data found. Please check the uploaded CSV files "
        "and the standardized data above."
    )
    st.stop()

st.success(f"Filters applied successfully: {len(view)} rows selected.")
# Algorithm selection — visible checkboxes instead of dropdown
with f1:
    st.markdown("### Select Algorithms")
    chosen_algorithms = []

    for algorithm in algorithms:
        selected = st.checkbox(
            algorithm,
            value=True,
            key=f"algorithm_{algorithm}"
        )
        if selected:
            chosen_algorithms.append(algorithm)

# Test-condition selection
with f2:
    st.markdown("### Select Test Conditions")
    chosen_conditions = []

    for condition in conditions:
        selected = st.checkbox(
            condition,
            value=True,
            key=f"condition_{condition}"
        )
        if selected:
            chosen_conditions.append(condition)

# Apply filters
view = normalized[
    normalized["Algorithm"].astype(str).isin(chosen_algorithms)
    & normalized["Environment"].astype(str).isin(chosen_conditions)
].copy()

if view.empty:
    st.warning("No rows match the selected filters. Choose at least one algorithm and one test condition.")
    st.stop()

metric = "Balanced Accuracy" if view["Balanced Accuracy"].notna().any() else "Accuracy"
if not view[metric].notna().any():
    st.warning("The selected rows do not contain numeric Accuracy or Balanced Accuracy values.")
    st.stop()

st.caption(f"Primary metric for environment comparisons: **{metric}**. Validation accuracy is shown separately where available.")

# Convert 0..1 metrics to percentages for display only; retain raw values in downloads.
def as_percent(series):
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().any() and vals.max(skipna=True) <= 1.0:
        return vals * 100
    return vals

view["Metric Display (%)"] = as_percent(view[metric])
view["Validation Display (%)"] = as_percent(view["Validation Accuracy"])

# ----------------------------- Summary metrics -------------------------------
normal_rows = view[view["Environment"] == "Normal Test"]
normal_mean = normal_rows["Metric Display (%)"].mean()
all_mean = view["Metric Display (%)"].mean()
val_mean = view["Validation Display (%)"].mean()

m1, m2, m3 = st.columns(3)
m1.metric("Mean validation score", f"{val_mean:.2f}%" if pd.notna(val_mean) else "Not available")
m2.metric("Mean normal-test score", f"{normal_mean:.2f}%" if pd.notna(normal_mean) else "Not available")
m3.metric("Mean selected-condition score", f"{all_mean:.2f}%" if pd.notna(all_mean) else "Not available")

# ----------------------------- Graphs ----------------------------------------
st.subheader("3. Performance graphs")
left, right = st.columns(2)

with left:
    st.markdown("**Mean performance by algorithm and condition**")
    plot_data = view.dropna(subset=["Metric Display (%)"]).groupby(["Algorithm", "Environment"], as_index=False)["Metric Display (%)"].mean()
    if not plot_data.empty:
        pivot = plot_data.pivot(index="Algorithm", columns="Environment", values="Metric Display (%)")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, max(100, float(pivot.max().max()) * 1.08))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Not enough numeric values to draw the comparison chart.")

with right:
    st.markdown("**Validation score by algorithm**")
    val_data = view.dropna(subset=["Validation Display (%)"]).groupby("Algorithm", as_index=False)["Validation Display (%)"].mean()
    if not val_data.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(val_data["Algorithm"], val_data["Validation Display (%)"])
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_ylim(0, max(100, float(val_data["Validation Display (%)"].max()) * 1.08))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Validation accuracy was not available in the uploaded files.")

# ----------------------------- Reliability analysis --------------------------
st.subheader("4. Reliability analysis")
st.write(
    "The dashboard compares recorded test scores with normal-test performance. "
    "A larger decrease under an altered condition can indicate sensitivity to that particular synthetic test. "
    "It does not, by itself, prove why a model changed."
)

grouped = view.dropna(subset=["Metric Display (%)"]).groupby(["Algorithm", "Configuration", "Environment"], as_index=False)["Metric Display (%)"].mean()
normal_lookup = grouped[grouped["Environment"] == "Normal Test"][["Algorithm", "Configuration", "Metric Display (%)"]].rename(columns={"Metric Display (%)": "Normal Test (%)"})
if not normal_lookup.empty:
    analysis = grouped.merge(normal_lookup, on=["Algorithm", "Configuration"], how="left")
    analysis["Performance change (percentage points)"] = analysis["Metric Display (%)"] - analysis["Normal Test (%)"]
    analysis["Performance drop (percentage points)"] = -analysis["Performance change (percentage points)"]
    analysis = analysis[analysis["Environment"] != "Normal Test"].copy()
    if not analysis.empty:
        def interpretation(row):
            drop = row["Performance drop (percentage points)"]
            if pd.isna(drop):
                return "Normal-test baseline unavailable"
            if drop > 5:
                return "Larger drop in this test"
            if drop > 0:
                return "Small drop in this test"
            if drop < 0:
                return "Score increased in this test"
            return "No observed change"
        analysis["Observation"] = analysis.apply(interpretation, axis=1)
        st.dataframe(
            analysis[["Algorithm", "Configuration", "Environment", "Normal Test (%)", "Metric Display (%)",
                      "Performance change (percentage points)", "Observation"]],
            use_container_width=True, hide_index=True
        )
        st.caption("The 5-percentage-point label is a descriptive threshold for this dashboard, not a universal scientific standard.")
    else:
        st.info("No altered test conditions were found alongside a normal-test baseline.")
else:
    st.info("A normal-test baseline was not detected. Rename the relevant environment to include 'Normal' or check the source CSV format.")

# Validation similarity / variation analysis
st.markdown("**Validation scores and test-score variation**")
variation = view.dropna(subset=["Metric Display (%)"]).groupby(["Algorithm", "Configuration"], as_index=False).agg(
    Mean_test_score=("Metric Display (%)", "mean"),
    Lowest_test_score=("Metric Display (%)", "min"),
    Highest_test_score=("Metric Display (%)", "max"),
    Conditions_tested=("Environment", "nunique"),
)
variation["Observed score range (percentage points)"] = variation["Highest_test_score"] - variation["Lowest_test_score"]
val_per_config = view.dropna(subset=["Validation Display (%)"]).groupby(["Algorithm", "Configuration"], as_index=False)["Validation Display (%)"].mean()
variation = variation.merge(val_per_config, on=["Algorithm", "Configuration"], how="left")
variation = variation.rename(columns={"Validation Display (%)": "Validation score (%)"})
st.dataframe(variation, use_container_width=True, hide_index=True)

st.subheader("5. Download results")
download_col1, download_col2 = st.columns(2)
with download_col1:
    combined_bytes = raw.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download combined uploaded data",
        data=combined_bytes,
        file_name="modelcheck_combined_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
with download_col2:
    summary_bytes = variation.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download model summary",
        data=summary_bytes,
        file_name="modelcheck_model_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.subheader("About the analysis")
st.markdown("""
- The app analyzes **recorded metrics** from uploaded CSV files; it does not train or alter models.
- It can show performance differences across recorded conditions, but cannot identify individual misclassified samples unless row-level true labels and predictions are provided.
- Synthetic noise and feature shifts are simplified demonstrations, not proof of real-world deployment behavior.
- Use the same split and evaluation protocol when comparing algorithms. Do not use the test set repeatedly to tune models.
""")
st.caption("ModelCheck AI · Research demonstration · Built with Streamlit, pandas and Matplotlib")
