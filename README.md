# Prediction-of-Diabetes


This repository contains a Jupyter notebook (`eda-preprocessing-model-training.ipynb`) that implements exploratory data analysis (EDA), preprocessing/feature engineering, and model training utilities for a classification/regression task (example: diabetes prediction). The notebook is structured as reusable Python classes so you can run the whole pipeline interactively or import the components into other scripts. This notebook ranks 18/4206 in the competition.

**Key Components**

- `Config`:
  - Central configuration (data paths, target name, device selection, CV folds, random seed, task type).
  - Loads datasets (original dataset and Kaggle train/test sample paths used in the notebook).

- `EDA`:
  - Multiple versions included; current version is a robust class that accepts overrides in the constructor.
  - Functions: `data_info()`, `heatmap()`, `dist_plots()`, `cat_feature_plots()`, `target_pie()`, `target_plot()`.
  - Automatically detects categorical and numerical features (excludes target column(s)).

- `Preprocessing`:
  - Prepares `X`, `y` and `test` splits, fills missing values if configured, feature engineering (merging aggregated stats from an `orig` dataset), category frequency encodings, and optional outlier removal / log transform.
  - Returns processed `X, y, test, cat_features, num_features` via `fit_transform()`.

- `Trainer`:
  - Training loop with cross-validation and fold handling.
  - Supports LightGBM, XGBoost, CatBoost, neural/tabular models (as used in the notebook), plus generic sklearn-compatible estimators.
  - Saves OOF and test predictions (`<model>_oof.csv`, `<model>_test.csv`) and can produce an ensemble meta-model.
  - Visualization of ROC, confusion matrix, or regression residuals depending on `task_type`.

**Data**

- Notebook uses Kaggle-style dataset paths by default. Example paths used in the notebook:
  - `/kaggle/input/playground-series-s5e12/train.csv`
  - `/kaggle/input/playground-series-s5e12/test.csv`
  - `/kaggle/input/playground-series-s5e12/sample_submission.csv`
  - `/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv`

- If you are running locally, either mount the same folder structure or update the paths in `Config` to point to your local CSV files.

**Requirements (suggested)**

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lightgbm (optional for LightGBM models)
- xgboost (optional for XGBoost models)
- catboost (optional for CatBoost models)
- torch (optional, used to detect device and for NN models if used)
- colorama
- ipython
- (Optional) `category_encoders` or a custom `TargetEncoder` implementation if the notebook expects it

Install with pip (example):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost torch colorama ipython
# optional: pip install category_encoders
```

**How to use**

1. Open the notebook `eda-preprocessing-model-training.ipynb` in Jupyter or JupyterLab.
2. Review `Config` to ensure data file paths and `target` are correct for your setup.
3. Run cells top-to-bottom to load data and execute EDA / preprocessing / training routines.

Quick example (in a notebook cell):

```python
# instantiate EDA with defaults from Config
eda = EDA()

# run preprocessing
prep = Preprocessing()
X, y, test, cat_features, num_features = prep.fit_transform()

# train models (example - models dict should be defined in the notebook)
trainer = Trainer(X, y, test, models=models, num_features=num_features, cat_features=cat_features)
preds = trainer.run()
```

**Outputs**

- Per-model OOF and TEST predictions are saved as `<model_name>_oof.csv` and `<model_name>_test.csv` in the working directory when training runs complete.
- If multiple models are provided, an ensemble meta-model is trained and its OOF/TEST predictions are produced.
- Visualizations: correlation heatmap, distribution plots, categorical bar plots, ROC/Confusion Matrix (classification) or residual plots (regression).

**Notes & Caveats**

- The notebook contains several versions of `EDA` and `Preprocessing` for iterative development—use the most recent `Version 3` classes for robustness.
- Some helper classes referenced (e.g., `TargetEncoder`, `FeatureEncoder`, `FeatureEncoder.transform_fold`) may be custom or require `category_encoders` — ensure these implementations are available in your environment or adapt as needed.
- Device selection uses `torch.cuda.is_available()`; PyTorch is optional unless using neural models.
- The notebook assumes the dataset has an `id` index in some CSVs; verify `index_col` usage when reading CSVs.

**Author / License**

- Author: notebook owner
- License: not specified (add your license file if needed)
