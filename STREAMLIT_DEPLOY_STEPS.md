# Deploy Battery SOH + RUL on Streamlit Community Cloud (free tier)

## Files
Upload the CONTENTS of this folder to the root of a GitHub repository. Keep `app.py`, `requirements.txt`, `models.py`, `pipeline.py`, `trend.py`, `charts.py`, `report.py`, and `rul_core.py` together at the repository root. Include `README.md` and `.streamlit/config.toml` too.

## GitHub
1. Create a repository at https://github.com/new (public is easiest for a public demo; do not upload private/customer battery data or secrets).
2. Upload the project files to the repository root and commit them to `main`.

## Streamlit
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Click Create app, select the repository and `main` branch.
3. Set main file path to `app.py`.
4. In advanced settings choose Python 3.11 if the option is available.
5. Deploy and wait for dependency installation to finish.

## Test
Start with a small representative CSV/XLSX, then test a larger file. The app trains models when Run pipeline is pressed, so large datasets or high LSTM epoch/ensemble settings may exceed Community Cloud's free CPU/RAM limits. Start with a small dataset and lower the epochs/ensemble size if needed. If the app fails, open Manage app → Logs and copy the complete error.

The CPU-only PyTorch dependency avoids installing CUDA-enabled packages on a CPU host. It still downloads a sizeable PyTorch wheel, so the initial build may take several minutes.
