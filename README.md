# ModelCheck AI — Model Reliability Analyzer

A Streamlit dashboard for comparing machine-learning experiment result CSVs across normal testing, synthetic noise, and feature-shift conditions.

## Features
- Upload multiple CSV files
- Standardize common result formats
- Compare Logistic Regression, SVM, Decision Tree, and Random Forest
- Plot validation and test performance
- Summarize observed score changes across test conditions
- Download combined uploaded data and a model summary

## Run locally (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Open Command Prompt and go to this folder. Example:
   `cd Desktop\ModelCheckAI`
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Run:
   `python -m streamlit run app.py`
5. Open the local address Streamlit prints, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud
1. Create/sign in to GitHub: https://github.com/
2. Create a **public** repository named `ModelCheckAI` (or another name).
3. Upload `app.py` and `requirements.txt` from this folder to the repository root.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select the repository, branch (usually `main`), and main file path `app.py`.
6. Click **Deploy**. Wait for the build to finish.
7. Open the public URL Streamlit gives you and test the upload, charts, and downloads.

If Streamlit asks for permissions, review and authorize the connection. Never upload passwords, API keys, or private data to a public repository.

## CSV format
The dashboard supports common tidy/long-format columns such as:
- `Algorithm`
- `Configuration`
- `Validation Accuracy` (optional)
- `Environment` (e.g. `Normal Test`, `Noise Test`, `Feature-Shift Test`)
- `Accuracy` and/or `Balanced Accuracy`

It also attempts to detect wide-format accuracy columns containing terms such as Normal, Noise, or Shift.

## Research integrity notes
- Use your actual experiment outputs for the research paper and presentation.
- Do not describe illustrative or synthetic example values as measured results.
- This dashboard reports metrics from CSVs; it does not retrain models or prove the cause of a performance change.
