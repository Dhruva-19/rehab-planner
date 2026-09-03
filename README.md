# AI Personalized Rehabilitation Planner

Final-year project: recognizes physiotherapy/rehab exercises from wearable
IMU sensor data (accelerometer + gyroscope), assesses movement quality, and
gives personalized feedback via a Streamlit dashboard.

## Project Structure
```
rehab_planner/
├── data/
│   ├── raw/            # untouched downloaded datasets (MM-Fit goes here)
│   └── processed/      # cleaned/windowed/feature-engineered data
├── notebooks/          # exploratory analysis, one notebook per stage
├── src/
│   ├── data_loader.py  # loads MM-Fit sessions into DataFrames
│   └── models/         # training + inference code
├── dashboard/          # Streamlit app
├── database/           # SQLite session history
├── saved_models/       # trained model artifacts (.pkl / .h5)
├── reference/           # official mm-fit starter repo (EDA notebook, docs)
├── config.py            # all paths & constants — import this, don't hardcode paths
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the MM-Fit Dataset

The dataset itself is **not on GitHub** — it's hosted separately:

1. Visit **https://mmfit.github.io/** and follow the download link there.
2. Extract the archive so you end up with 21 session folders (`w00` … `w20`)
   sitting directly inside `data/raw/mm-fit/`, e.g.:
   ```
   data/raw/mm-fit/w00/w00_sw_r_acc.npy
   data/raw/mm-fit/w00/w00_sw_r_gyr.npy
   data/raw/mm-fit/w00/w00_labels.csv
   ...
   ```
3. Run the smoke test:
   ```bash
   python src/data_loader.py
   ```
   This should print the number of sessions found and an activity
   distribution table. If you see a `FileNotFoundError`, double-check the
   folder path above.

### Dataset facts (from the official docs)
- 10 participants, 21 recorded sessions (`w00`–`w20`), 10 exercises +
  a "non_activity" background class.
- We use the **right smartwatch accelerometer + gyroscope**
  (`sw_r_acc`, `sw_r_gyr`) by default, since a single wrist-worn device
  is the most realistic stand-in for a home rehab wearable. Sampling
  rate is 100 Hz.
- The dataset authors' recommended train/val/test/unseen_test split is
  already encoded in `config.py` as `MMFIT_SPLIT` — use `unseen_test`
  as your final held-out cross-subject evaluation.
- `reference/mm-fit/` contains the official starter repo (EDA notebook +
  helper functions) if you want to cross-check anything.

## Fallback Datasets
If MM-Fit access is ever an issue, PAMAP2 and MHEALTH are the agreed
fallbacks (see project knowledge) — both are on the UCI ML Repository.

## Workflow (5 stages, ~12–14 weeks)
1. Dataset selection ✅ (MM-Fit)
2. Preprocessing & feature engineering — **next**
3. Exercise recognition modeling (Random Forest / XGBoost baseline, optional LSTM)
4. Rule-based / unsupervised quality assessment
5. Feedback engine + Streamlit dashboard + SQLite session history
