# EEG-Based Motor Imagery Classification for BCI Applications

> **Fernando Sala-Vivé, Raid Homs, Anton Tenev, Marcos Oriol**  
> Master of Neuroengineering and Rehabilitation — Universitat Politècnica de Catalunya (UPC)

A complete machine learning pipeline for classifying three motor imagery conditions — **left-hand**, **right-hand**, and **rest** — from 16-channel EEG recordings. The pipeline covers signal preprocessing, feature extraction (ERD + MRCP + CSP), and supervised classification evaluated under both within-subject stratified holdout (80/20 split) and Leave-One-Subject-Out (LOSO) cross-validation.

---

## Results at a glance

| Pipeline | Within-subject | LOSO |
|---|---|---|
| Model 1 — ERD + MRCP (Cz, Fz) | 46.6 ± 8.4% | 42.6 ± 7.4% |
| Model 2 — ERD + MRCP (C3, Cz, C4, Fz) | 48.4 ± 9.9% | 44.5 ± 8.5% |
| **CSP cascade (8ch, Mu+Beta)** | **59.0 ± 8.9%** | **49.4 ± 8.9%** |

Chance level: 33.3% (3 balanced classes). Pipeline computational cost: < 1.3 ms (excluding 1500 ms acquisition window).

---

## Scripts

**`Trabajo_final.m`** — MATLAB preprocessing and feature engineering script. Runs on all 11 subjects and produces the CSV and .mat files consumed by the Python notebook. Covers:
- Signal detrending and bandpass filtering (ERD 8–30 Hz, Mu 8–13 Hz, Beta 14–30 Hz, MRCP 0.3–3 Hz) using 4th-order Butterworth filters
- Automatic bad channel detection and Common Average Reference (CAR)
- Adaptive trial rejection based on the 95th percentile of peak amplitude and standard deviation across C3, Cz, C4
- ERD feature extraction (band power relative to baseline) and MRCP temporal descriptors (min, max, AUC, std) for Models 1 and 2
- Grand-average topographic maps via EEGLab `topoplot` for Mu-ERD, Beta-ERD, and MRCP
- Kruskal-Wallis and Cohen's d feature discriminability analysis
- Export to `features_model1_all.csv`, `features_model2_all.csv`, and `trials_CSP_all.mat`

**`Trabajo_Final_v3.ipynb`** — Python classification and evaluation notebook. Loads the MATLAB outputs and runs:
- Within-subject (80/20 stratified split) and LOSO evaluation for Models 1 and 2 across LDA+PCA, SVM, Logistic Regression, and Random Forest
- CSP cascade architecture: CSP1 (rest vs. movement) → CSP2 (left vs. right), evaluated across 3-channel and 8-channel configurations and four frequency band combinations (Mu, Beta, Mu+Beta, full ERD), using CSP+LDA+SelectKBest, CSP+LDA, CSP+SVM, and CSP+LogReg
- Confusion matrices, per-class precision/recall/F1, per-subject LOSO accuracy plots
- Algorithmic latency benchmarking (t_window + t_preprocessing + t_inference)
- Export of all results to CSV

---

## Pipeline overview

```
Raw EEG (16 ch, 125 Hz)
        │
        ▼
  Detrend + Bandpass filters (ERD / Mu / Beta / MRCP)
        │
        ▼
  Bad channel detection + CAR rereferencing
        │
        ▼
  Epoch segmentation + adaptive trial rejection (95th pct)
        │
        ├──────────────────────┬────────────────────────
        │                      │
   ERD + MRCP features     CSP trials (3ch / 8ch)
   (Models 1 & 2)          (Mu / Beta / Mu+Beta / full)
        │                      │
        ▼                      ▼
  LDA+PCA / SVM /          CSP cascade:
  LogReg / RF              Stage 1: rest vs. movement
  within-subject + LOSO    Stage 2: left vs. right
                           Classifiers: CSP+LDA+SelectKBest,
                           CSP+LDA, CSP+SVM, CSP+LogReg
                                │
                                ▼
                         within-subject + LOSO + latency
```

---

## Data format

The dataset is publicly available at:  
**[https://drive.google.com/drive/u/1/folders/1W--2vaI5t-k_xI-eTj6Rx6SmvPK_g67v](https://drive.google.com/drive/u/1/folders/1W--2vaI5t-k_xI-eTj6Rx6SmvPK_g67v)**

The pipeline expects one `.mat` file per subject named `v01.mat` … `v11.mat`, each containing an EEGLab-compatible `EEG` struct with fields:
- `EEG.data` — raw EEG matrix `(channels × samples)`
- `EEG.srate` — sampling frequency (125 Hz)
- `EEG.event` — event struct array with `.latency` and `.type` fields
  - Event types: `'l'` (left), `'r'` (right), `'b'` (rest/bilateral)
- Channel electrode location file: `coord_15elecMotorImagery.loc` (EEGLab format, required for topographic maps)

The first 120 events correspond to motor execution runs (skipped); motor imagery trials start at event index 121.

---

## Feature descriptions

### ERD features (8 features per trial)

Computed over the 0.5–1.5 s post-cue window from channels C3, Cz, C4:

| Feature | Description |
|---|---|
| `mu_C3`, `mu_Cz`, `mu_C4` | Mu-band (8–13 Hz) ERD relative to 2 s pre-cue baseline |
| `beta_C3`, `beta_Cz`, `beta_C4` | Beta-band (14–30 Hz) ERD relative to baseline |
| `lat_mu` | Mu lateralization index: C3 − C4 |
| `lat_beta` | Beta lateralization index: C3 − C4 |

### MRCP features

Computed over the −0.5 to +0.5 s window around cue onset:

| Feature | Description |
|---|---|
| `mrcp_min`, `mrcp_max` | Minimum and maximum amplitude |
| `mrcp_auc` | Area under the curve (via `trapz`) |
| `mrcp_std` | Standard deviation |

**Model 1** uses Cz and Fz (8 MRCP features). **Model 2** uses C3, Cz, C4, and Fz (16 MRCP features).

### CSP trials

Raw epoched trials stored as `(channels × samples × trials)` arrays in `trials_CSP_all.mat`, with channel configurations:
- 3-channel: C3, Cz, C4
- 8-channel: C3, Cz, C4, P3, P4, F3, F4, Fz

---

## Classifiers

### Models 1 & 2 (ERD + MRCP flat features)

| Classifier | Notes |
|---|---|
| LDA | `solver='lsqr'`, `shrinkage='auto'`, no dimensionality reduction |
| LDA + PCA | PCA retains 95% of variance before LDA. Used to handle high intra-block ERD correlation. |
| SVM | RBF kernel, `C=1`, `gamma=0.1`, balanced class weights |
| Logistic Regression | `max_iter=1000`, balanced class weights |
| Random Forest | 100 estimators |

All classifiers receive `StandardScaler`-normalized features. Both LDA and LDA+PCA are evaluated independently — LDA+PCA addresses the high inter-electrode correlation in the ERD block, while plain LDA is included as a simpler baseline.

### CSP cascade (both stages)

Each stage of the cascade independently selects the best classifier from:

| Classifier | Pipeline |
|---|---|
| CSP + LDA + SelectKBest | CSP spatial filtering → SelectKBest (`f_classif`, k=4) → LDA |
| CSP + LDA | CSP spatial filtering → LDA (`solver='lsqr'`, `shrinkage='auto'`) |
| CSP + SVM | CSP spatial filtering → StandardScaler → SVM (RBF) |
| CSP + LogReg | CSP spatial filtering → StandardScaler → Logistic Regression |

All CSP models use Ledoit-Wolf regularization (`reg='ledoit_wolf'`) and `n_components=8`. Trials are per-channel variance-normalized before CSP. The best-performing classifier is selected independently for each stage and each subject fold.

---

## Requirements

### MATLAB
- MATLAB R2020b or later
- Signal Processing Toolbox (for `butter`, `filtfilt`, `bandpower`)
- EEGLab (for `topoplot` and `readlocs`)

### Python
```
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
mne
```

Install Python dependencies:
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn mne
```

---

## Running the pipeline

**Step 1 — MATLAB preprocessing**

Place all subject `.mat` files and `coord_15elecMotorImagery.loc` in the same directory as `Trabajo_final.m`, then run:

```matlab
run('Trabajo_final.m')
```

Outputs generated:
- `features_model1_all.csv` — ERD + MRCP features for Model 1 (all subjects)
- `features_model2_all.csv` — ERD + MRCP features for Model 2 (all subjects)
- `trials_CSP_all.mat` — epoched trials for CSP classification

**Step 2 — Python classification**

Place the three output files in the same directory as the notebook and run all cells:

```bash
jupyter notebook Trabajo_Final_v3.ipynb
```

Results exported:
- `results_loso_all.csv` — per-subject LOSO accuracy for all models
- `results_within_all.csv` — per-subject within-subject accuracy for all models
- `results_summary.csv` — best configuration summary table

---

## Key design decisions

**Why no ICA for artifact removal?**  
ICA requires the number of data points to substantially exceed N² (where N is the number of channels) for stable decomposition. With only 16 channels the spatial mixing matrix is poorly conditioned and source separation is unreliable. Adaptive threshold-based trial rejection was used instead, achieving a mean rejection rate of 4.2 ± 0.9% across subjects.

**Why a cascade architecture for CSP?**  
CSP is inherently binary. Feature analysis revealed that MRCP features best discriminate rest from movement, while Mu-band ERD lateralization best discriminates left from right — two complementary feature families for two separate binary problems. The cascade makes this decomposition explicit. The trade-off is that first-stage errors propagate to the second stage, capping overall performance at 59.0 ± 8.9% within-subject.

**Why 1500 ms latency is a limitation**  
The ERD feature extraction window (0.5–1.5 s post-cue) imposes a fixed 1500 ms acquisition delay that dominates total system latency (1501.3 ms), well above the 500 ms threshold for responsive closed-loop BCI operation. Processing cost is negligible (<1.3 ms). A sliding-window approach with shorter epochs (~500 ms) would reduce latency to 100–200 ms at the cost of spectral estimation accuracy.

---

## Reference

If you use or build on this work, please cite the accompanying paper:

> F. Sala-Vivé, R. Homs, A. Tenev, M. Oriol. *EEG-Based Motor Imagery Classification Using Machine Learning Techniques for Brain-Computer Interface Applications.* Master of Neuroengineering and Rehabilitation, UPC, 2025.

---

## License

This repository is shared for academic and educational purposes. Please contact the authors before reusing code or data in published work.
