"""
Generate calibrated out-of-fold probabilities and preliminary reliable-negative
(RN) samples from a feature dataset prior to PU-learning.

This module loads an account-level feature table, applies multiple base
classifiers (Random Forest, Logistic Regression, and optionally Linear SVM),
and produces calibrated out-of-fold (OOF) probabilities using Stratified
K-fold cross-validation. These OOF scores serve as intermediate inputs for
downstream PU-learning pipelines.

For each configured random seed, the workflow performs:
    1. Construction of preprocessing pipelines with imputation and scaling.
    2. Model training on each fold and probability calibration via Platt
       scaling or isotonic regression.
    3. Generation of per-model OOF probabilities and saving them to disk.
    4. Extraction of preliminary reliable-negative (RN) candidates by selecting
       unlabeled samples whose predicted probabilities fall below a fixed
       threshold across all models.
    5. Optional conservative RN filtering, enforcing additional constraints on
       probability stability and distance to positive samples.
    6. Output of per-seed RN lists, positive-below-threshold diagnostics, and
       conservative RN lists when enabled.

When multiple seeds are specified, the module also merges multi-seed OOF
probabilities and composes final RN consensus sets using intersection or
majority voting. These outputs constitute the intermediate feature-selection
stage before the final PU-learning and classifier training steps.
"""
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Adjustable parameters (can be modified as needed)
INPUT_CSV = "Feature.csv"
LABEL_COL = "is_alert"         # True = P, False = U
ID_COLS   = ["acct", "ID", "acct_id", "account"]

# We use Random Forest, Logistic regression, Support Vector Machine
USE_MODELS = ["RF", "LR", "SVM"]

# K-fold, calibration method and threshold
K_FOLDS     = 5                     # 5 or 10
CALIBRATOR  = "sigmoid"             # 'sigmoid' or 'isotonic'
TAU         = 0.04                  # RN threshold: the probability should less than TAU for all models .

# Multiple sub-configurations (allowing for multiple stability enhancement runs)
SEEDS = [495, 2485, 95265]

# RN consensus mode: 'intersection' or 'majority'
RN_CONSENSUS_MODE = "intersection"
RN_MIN_VOTES      = None           # The minimum number of votes required in majority mode; if no votes are given, take the ceil(#runs/2)

# Hyperparameters for RF/LR/SVM (conservative safety)
RF_PARAMS = dict(
    n_estimators=1200,
    max_depth=12,
    min_samples_leaf=8,
    min_samples_split=5,
    max_features="sqrt",
    bootstrap=True,
    max_samples=None,
    criterion="gini",
    class_weight="balanced_subsample",
    n_jobs=-1,
)
LR_PARAMS = dict(
    C=0.1,
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
)
SVM_PARAMS = dict(
    C=0.1,
    loss="squared_hinge",
    penalty="l2",
    dual=True,
    tol=1e-3,
    max_iter=8000,
    class_weight="balanced",
)

# Conservative RN goalkeeping parameters (adjustable)
CONSERVATIVE_RN = True
TAU_MEAN_FACTOR = 0.7   # mean_prob < TAU * 0.7
SIGMA_MAX       = 0.02  # Upper limit of the standard deviation of prob for each model (consistency required)
POS_GUARD_PERCENTILE = 20.0  # Quantile threshold for the nearest neighbor distance (for positive classes, the distance must be greater than this threshold).

# Ray Tune Parameter
USE_RAY_TUNE = True      # If you wants to use Ray Tune, set to True
TUNE_METRIC  = "roc_auc"       # "ap" = average precision；或 "roc_auc"
TUNE_KFOLDS  = 3          # K_FOLDS
TUNE_NUM_SAMPLES = 36     # Sample number
TUNE_MAX_CONCURRENT_TRIALS = 4

def load_data(input_csv: str, label_col: str, id_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load feature table and split into features, labels, and IDs.

    Reads a CSV file, validates that the label column exists, chooses an
    identifier column if available, coerces the label to integers, converts
    all remaining columns to numeric features, and returns (X, y, ids).
    """
    df = pd.read_csv(input_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {input_csv}.")

    id_candidates = [c for c in id_cols if c in df.columns]
    if len(id_candidates) > 0:
        ids = df[id_candidates[0]].astype(str)
    else:
        ids = pd.Series(df.index.astype(str), name="row_id")

    # Prepare X、y
    y = df[label_col].astype(int)  # True/False -> 1/0
    drop_cols = [label_col] + id_candidates
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    # All features are attempted to be converted to numerical values; those that cannot be converted are set to NaN.
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, ids


def build_model_pipelines(random_state: int) -> dict[str, Pipeline]:
    """
    Build preprocessing and estimator pipelines for enabled model families.

    Creates scikit-learn Pipelines for RandomForest, LogisticRegression, and
    LinearSVC (depending on USE_MODELS) using median imputation and optional
    standardization. Returns a dictionary keyed by short model name.
    """
    models: dict[str, Pipeline] = {}

    if "RF" in USE_MODELS:
        rf = RandomForestClassifier(random_state=random_state, **RF_PARAMS)
        models["RF"] = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", rf),
        ])

    if "LR" in USE_MODELS:
        lr = LogisticRegression(random_state=random_state, **LR_PARAMS)
        models["LR"] = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", lr),
        ])

    if "SVM" in USE_MODELS:
        svm = LinearSVC(random_state=random_state, **SVM_PARAMS)
        models["SVM"] = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", svm),
        ])

    if len(models) == 0:
        raise ValueError("No models selected. Please set USE_MODELS to include at least one of ['RF','LR','SVM'].")
    return models

def _calibrated_cv_fit_predict(estimator: Pipeline, X_tr, y_tr, X_va, calibrator: str) -> np.ndarray:
    """
    Fit a calibrated wrapper around an estimator and predict probabilities.

    Wraps the given pipeline in CalibratedClassifierCV using the requested
    calibration method, handles both `estimator` and `base_estimator`
    argument names for scikit-learn version compatibility, and returns
    calibrated P(y = 1) on the validation data.
    """
    try:
        cal = CalibratedClassifierCV(estimator=estimator, method=calibrator, cv=5)
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=estimator, method=calibrator, cv=5)
    cal.fit(X_tr, y_tr)
    proba = cal.predict_proba(X_va)[:, 1]
    return proba

def oof_calibrated_probs(
    X: pd.DataFrame, y: pd.Series, model_name: str, base_pipeline: Pipeline,
    k_folds: int, calibrator: str, seed: int
) -> np.ndarray:
    """
    Compute calibrated out-of-fold probabilities for a single model.

    Runs StratifiedKFold cross-validation, fits a cloned pipeline on each
    training fold, obtains calibrated probabilities on the corresponding
    validation fold, and assembles a full-length OOF probability vector
    aligned with the original sample order.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        est = clone(base_pipeline)
        proba = _calibrated_cv_fit_predict(est, X_tr, y_tr, X_va, CALIBRATOR)
        oof[va] = proba

        print(f"[seed={seed}] {model_name} | fold {fold}/{k_folds} done.")
    return oof


def run_one_seed(
    X: pd.DataFrame, y: pd.Series, ids: pd.Series,
    seed: int, k_folds: int, calibrator: str, tau: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the OOF and RN-selection workflow for a single random seed.

    For each enabled model family, this function computes calibrated OOF
    probabilities, stacks them into a DataFrame, and derives three views:

    * `oof_df` – full OOF table with one row per sample and one column per
      model probability.
    * `rn_df` – reliable negative candidates, i.e. unlabeled samples whose
      probabilities from all models are strictly below `tau`.
    * `poslow_df` – labeled positive samples whose probabilities from all
      models are below `tau` (useful as diagnostics for overly confident
      negative predictions).
    """
    model_pipes = build_model_pipelines(random_state=seed)

    # collect OOF
    oof_dict: dict[str, np.ndarray] = {}
    for name, pipe in model_pipes.items():
        oof_dict[name] = oof_calibrated_probs(
            X=X, y=y, model_name=name, base_pipeline=pipe,
            k_folds=k_folds, calibrator=calibrator, seed=seed
        )

    # OOF DataFrame
    oof_df = pd.DataFrame({"id": ids, "is_alert": y.astype(bool)})
    for name, arr in oof_dict.items():
        oof_df[f"{name}_oof_prob"] = arr

    # Threshold determination: All models's prob < tau
    all_below = np.ones(len(X), dtype=bool)
    for name in oof_dict:
        all_below &= (oof_dict[name] < tau)

    unlabeled_mask = (y.values == 0)           # U
    positive_mask  = (y.values == 1)           # P

    rn_mask      = unlabeled_mask & all_below  # Reliable negative samples
    poslow_mask  = positive_mask  & all_below  # Positive samples but below tau

    oof_df["all_models_below_tau"] = all_below
    oof_df["is_reliable_negative"] = rn_mask
    oof_df["is_pos_below_tau"]     = poslow_mask

    rn_df = oof_df.loc[rn_mask,   ["id"]].rename(columns={"id": "reliable_negative_id"}).reset_index(drop=True)
    poslow_df = oof_df.loc[poslow_mask, ["id"]].rename(columns={"id": "pos_below_tau_id"}).reset_index(drop=True)

    return oof_df, rn_df, poslow_df

def pick_rn_conservative(
    oof_df: pd.DataFrame,
    y: pd.Series,
    X_features: pd.DataFrame,
    model_cols: list[str],
    tau_rn: float = TAU,
    tau_mean_factor: float = TAU_MEAN_FACTOR,
    sigma_max: float = SIGMA_MAX,
    pos_guard_percentile: float = POS_GUARD_PERCENTILE,
    positive_label: int = 1,
    unlabeled_label: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Select a conservative subset of reliable negatives using multiple gates.

    Starting from the OOF probability table, this function filters unlabeled
    samples that satisfy:

    * all model probabilities are below `tau_rn`,
    * the mean probability across models is below `tau_mean_factor * tau_rn`,
    * the standard deviation of probabilities is below `sigma_max`,
    * the distance to the nearest positive sample in a reduced feature space
      exceeds a percentile threshold (`pos_guard_percentile`).

    The feature distance is computed after a standardization and dimensionality
    reduction step. The function returns a DataFrame with one column
    `reliable_negative_id` and a dictionary summarizing key statistics of the
    selection.
    """
    tau_mean = tau_rn * tau_mean_factor

    is_u = (y.values == unlabeled_label)
    probs = oof_df[model_cols].to_numpy()
    max_prob = probs.max(axis=1)
    mean_prob = probs.mean(axis=1)
    std_prob = probs.std(axis=1)

    # probability threshold + consistency
    mask_prob = (max_prob < tau_rn) & (std_prob < sigma_max) & (mean_prob < tau_mean)

    scaler = StandardScaler()
    Z = scaler.fit_transform(X_features.values)

    pos_idx = np.where(y.values == positive_label)[0]
    if len(pos_idx) < 2:
        mask_dist = np.ones(len(y), dtype=bool)
        delta = np.nan
    else:
        nn_pos = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn_pos.fit(Z[pos_idx])
        d_to_pos, _ = nn_pos.kneighbors(Z, return_distance=True)
        d_to_pos = d_to_pos.ravel()

        nn_pp = NearestNeighbors(n_neighbors=2, n_jobs=-1)
        nn_pp.fit(Z[pos_idx])
        d_pp, _ = nn_pp.kneighbors(Z[pos_idx], return_distance=True)
        d_pp = d_pp[:, 1]
        delta = np.percentile(d_pp, pos_guard_percentile)

        mask_dist = (d_to_pos > delta)

    rn_mask = is_u & mask_prob & mask_dist
    rn_df = oof_df.loc[rn_mask, ["id"]].copy().rename(columns={"id": "reliable_negative_id"}).reset_index(drop=True)

    report = {
        "U_candidates": int(is_u.sum()),
        "tau_rn": float(tau_rn),
        "tau_mean": float(tau_mean),
        "sigma_max": float(sigma_max),
        "pos_guard_percentile": float(pos_guard_percentile),
        "delta_pospos": float(delta) if not np.isnan(delta) else None,
        "picked_RN": int(rn_mask.sum()),
        "rejection_stats": {
            "prob_gate_fail": int((is_u & ~mask_prob).sum()),
            "dist_gate_fail": int((is_u & mask_prob & ~mask_dist).sum()),
        }
    }
    return rn_df, report

def consensus_from_runs(rn_list: list[pd.DataFrame], mode: str = "intersection", min_votes: int | None = None) -> pd.DataFrame:
    """
    Combine RN lists from multiple runs into a consensus set.

    Converts each input DataFrame into a set of IDs and aggregates them using
    either intersection (IDs present in all runs) or majority voting
    (IDs appearing in at least `min_votes` runs; defaults to half the number
    of runs rounded up). Returns a DataFrame with a single column
    `reliable_negative_id`.
    """
    if len(rn_list) == 0:
        return pd.DataFrame(columns=["reliable_negative_id"])

    if mode not in ("intersection", "majority"):
        raise ValueError("mode must be 'intersection' or 'majority'")

    sets = [set(df["reliable_negative_id"].astype(str)) for df in rn_list]

    if mode == "intersection":
        rn_final = set.intersection(*sets) if len(sets) > 1 else sets[0]
    else:
        if min_votes is None:
            min_votes = math.ceil(len(sets) / 2)
        from collections import Counter
        cnt = Counter()
        for s in sets:
            cnt.update(s)
        rn_final = {rid for rid, c in cnt.items() if c >= min_votes}

    return pd.DataFrame(sorted(rn_final), columns=["reliable_negative_id"])

def _maybe_import_ray():
    """
    Try to import Ray Tune components if available.

    Attempts to import `ray`, `tune`, `Tuner`, and `ASHAScheduler`. On
    success, returns these objects as a tuple. On failure, prints a short
    diagnostic message and returns a tuple of Nones so that the caller can
    gracefully skip tuning.
    """
    try:
        import ray
        from ray import tune
        from ray.tune import Tuner
        from ray.tune.schedulers import ASHAScheduler
        return ray, tune, Tuner, ASHAScheduler
    except Exception as e:
        print(f"[RayTune] Not available or failed to import: {e}")
        return None, None, None, None


def _model_from_params(model_name: str, random_state: int, params: dict) -> Pipeline:
    """
    Construct a model pipeline from a parameter dictionary.

    Rebuilds a RandomForest, LogisticRegression, or LinearSVC pipeline using
    the provided hyperparameters while preserving the preprocessing steps
    (imputation and optional scaling) used in the main workflow.
    """
    if model_name == "RF":
        rf = RandomForestClassifier(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", rf)])
    elif model_name == "LR":
        lr = LogisticRegression(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("lr", lr)])
    elif model_name == "SVM":
        svm = LinearSVC(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("svm", svm)])
    else:
        raise ValueError(model_name)


def _tune_space_for(model_name: str):
    """
    Define the Ray Tune search space for the given model family.

    Returns a configuration dictionary describing the hyperparameter
    distributions to explore for the specified model type (RF, LR, or SVM),
    using ranges and choices that are compatible with the default *_PARAMS
    settings in this module.
    """
    from ray import tune
    if model_name == "RF":
        return {
            "n_estimators": tune.randint(400, 1501),
            "max_depth": tune.choice([None, 8, 12, 16, 24]),
            "min_samples_leaf": tune.randint(2, 11),
            "min_samples_split": tune.choice([2, 5, 10, 20]),
            "max_features": tune.choice(["sqrt", "log2", 0.5, 0.8]),
            "bootstrap": True,
            "max_samples": tune.choice([None, 0.7, 0.9]),
            "criterion": tune.choice(["gini", "entropy"]),
            "class_weight": tune.choice(["balanced_subsample", "balanced"]),
            "n_jobs": -1
        }
    elif model_name == "LR":
        return {
            "C": tune.loguniform(1e-3, 1e1),
            "penalty": "l2",
            "solver": tune.choice(["lbfgs", "newton-cg"]),
            "max_iter": tune.choice([1000, 2000, 4000]),
            "class_weight": "balanced"
        }
    elif model_name == "SVM":
        return {
            "C": tune.loguniform(1e-2, 1e1),
            "loss": "squared_hinge",
            "penalty": "l2",
            "dual": True,
            "tol": tune.choice([1e-4, 1e-3]),
            "max_iter": tune.choice([4000, 8000, 12000]),
            "class_weight": "balanced"
        }
    else:
        raise ValueError(model_name)

def _score_oof(y_true: np.ndarray, oof_prob: np.ndarray, metric: str = "ap") -> float:
    """
    Score OOF probabilities against ground truth using a chosen metric.

    Supports `"ap"` for average precision and `"roc_auc"` for ROC AUC.
    The selected metric value is computed and returned as a floating-point
    number.
    """
    if metric == "ap":
        return float(average_precision_score(y_true, oof_prob))
    elif metric == "roc_auc":
        return float(roc_auc_score(y_true, oof_prob))
    else:
        raise ValueError("metric must be 'ap' or 'roc_auc'")

def ray_tune_models(X: pd.DataFrame, y: pd.Series, base_seed: int = 2025) -> dict:
    """
    Run Ray Tune hyperparameter search for the enabled models.

    For each model name listed in USE_MODELS, this function:

    * builds a Ray Tune search space via `_tune_space_for`,
    * defines a trainable that runs K-fold OOF estimation with calibration
      using the current configuration,
    * evaluates the configuration with `_score_oof` and reports additional
      metrics, and
    * uses an ASHAScheduler-backed Tuner to maximize the chosen score.

    The best configuration for each model is collected into a dictionary
    mapping model name to its tuned hyperparameters. If Ray is unavailable,
    an empty dictionary is returned and tuning is skipped.
    """
    ray, tune, Tuner, ASHAScheduler = _maybe_import_ray()
    if ray is None:
        print("[RayTune] Skip tuning (Ray not found).")
        return {}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    results = {}
    for model_name in USE_MODELS:
        space = _tune_space_for(model_name)

        def trainable(config):
            seed = base_seed
            pipe = _model_from_params(model_name, random_state=seed, params=config)
            skf = StratifiedKFold(n_splits=TUNE_KFOLDS, shuffle=True, random_state=seed)
            oof = np.zeros(len(X), dtype=float)
            for tr, va in skf.split(X, y):
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                y_tr = y.iloc[tr]
                proba = _calibrated_cv_fit_predict(pipe, X_tr, y_tr, X_va, CALIBRATOR)
                oof[va] = proba
            score = _score_oof(y.values, oof, metric=TUNE_METRIC)
            ap = average_precision_score(y.values, oof)
            roc = roc_auc_score(y.values, oof)
            from ray.air import session
            session.report({"score": float(score), "ap": float(ap), "roc_auc": float(roc)})

        scheduler = ASHAScheduler(grace_period=3, reduction_factor=2)

        tuner = Tuner(
            trainable,
            param_space=space,
            tune_config=tune.TuneConfig(
                metric="score",
                mode="max",
                num_samples=TUNE_NUM_SAMPLES,
                scheduler=scheduler,
                max_concurrent_trials=TUNE_MAX_CONCURRENT_TRIALS,
            ),
        )

        print(f"[RayTune] Tuning {model_name} ...")
        result = tuner.fit()
        best = result.get_best_result(metric="score", mode="max")
        best_conf = best.config
        print(f"[RayTune] Best {model_name} score={best.metrics['score']:.5f} | config={best_conf}")
        results[model_name] = best_conf
    try:
        ray.shutdown()
    except Exception:
        pass
    return results


def _apply_tuned_params(tuned: dict):
    """
    Update global model-parameter dictionaries with tuned configurations.

    For each model family present in the `tuned` mapping, this function
    overwrites matching keys in RF_PARAMS, LR_PARAMS, or SVM_PARAMS with the
    tuned values while leaving any unspecified keys at their defaults.
    """
    global RF_PARAMS, LR_PARAMS, SVM_PARAMS
    if "RF" in tuned:
        RF_PARAMS.update({k: v for k, v in tuned["RF"].items() if k in RF_PARAMS})
    if "LR" in tuned:
        LR_PARAMS.update({k: v for k, v in tuned["LR"].items() if k in LR_PARAMS})
    if "SVM" in tuned:
        SVM_PARAMS.update({k: v for k, v in tuned["SVM"].items() if k in SVM_PARAMS})

def main():
    """
    Entry point for the OOF and reliable-negative extraction pipeline.

    Loads the feature dataset, optionally runs Ray Tune to adjust model
    hyperparameters, and then, for each random seed specified in SEEDS:

    * builds model pipelines,
    * computes calibrated OOF probabilities,
    * selects preliminary reliable negatives and positives below threshold,
    * optionally applies conservative RN filtering, and
    * writes all per-seed outputs to CSV files.

    When multiple seeds are used, the function also produces merged OOF
    scores and consensus RN lists (and their conservative versions) across runs.
    """
    print(">>> Loading data...")
    X, y, ids = load_data(INPUT_CSV, LABEL_COL, ID_COLS)
    print(f"Data shape: X={X.shape}, y={y.shape}, positives={int(y.sum())}, unlabeled={int((y==0).sum())}")

    if USE_RAY_TUNE:
        print("\n>>> Ray Tune hyperparameter search (pre-run) ...")
        tuned = ray_tune_models(X, y, base_seed=2025)
        if tuned:
            _apply_tuned_params(tuned)
            print("[RayTune] Applied tuned params:")
            if "RF" in tuned: print("  RF_PARAMS:", RF_PARAMS)
            if "LR" in tuned: print("  LR_PARAMS:", LR_PARAMS)
            if "SVM" in tuned: print("  SVM_PARAMS:", SVM_PARAMS)
        else:
            print("[RayTune] No tuned params applied (Ray unavailable or tuning skipped).")

    all_oof_frames: list[pd.DataFrame] = []
    all_rn_frames:  list[pd.DataFrame] = []
    all_rn_cons_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        print(f"\n=== RUN seed={seed} | K={K_FOLDS} | calibrator={CALIBRATOR} | tau={TAU} ===")
        oof_df, rn_df, poslow_df = run_one_seed(
            X=X, y=y, ids=ids, seed=seed, k_folds=K_FOLDS, calibrator=CALIBRATOR, tau=TAU
        )

        # Save every round's result.
        oof_path    = f"oof_scores_seed{seed}.csv"
        rn_path     = f"reliable_negatives_seed{seed}.csv"
        poslow_path = f"positives_below_tau_seed{seed}.csv"
        oof_df.to_csv(oof_path, index=False)
        rn_df.to_csv(rn_path, index=False)
        poslow_df.to_csv(poslow_path, index=False)
        print(f"Saved: {oof_path}")
        print(f"Saved: {rn_path} | RN count: {len(rn_df)}")
        print(f"Saved: {poslow_path} | Pos<tau count: {len(poslow_df)}")

        if CONSERVATIVE_RN:
            model_cols = [c for c in oof_df.columns if c.endswith("_oof_prob")]
            rn_cons_df, rn_report = pick_rn_conservative(
                oof_df=oof_df,
                y=y,
                X_features=X,
                model_cols=model_cols,
                tau_rn=TAU,
                tau_mean_factor=TAU_MEAN_FACTOR,
                sigma_max=SIGMA_MAX,
                pos_guard_percentile=POS_GUARD_PERCENTILE
            )
            rn_cons_path = f"reliable_negatives_seed{seed}_conservative.csv"
            rn_cons_df.to_csv(rn_cons_path, index=False)
            print(f"Saved conservative RN: {rn_cons_path} | count={len(rn_cons_df)}")
            print("RN selection report:", rn_report)
            all_rn_cons_frames.append(rn_cons_df)

        oof_seed = oof_df.copy()
        for c in list(oof_seed.columns):
            if c.endswith("_oof_prob"):
                oof_seed.rename(columns={c: f"{c}_s{seed}"}, inplace=True)
        all_oof_frames.append(oof_seed)
        all_rn_frames.append(rn_df)

    if len(SEEDS) > 1:
        print("\n>>> Merging multi-seed OOF and building consensus RN ...")
        merged = all_oof_frames[0]
        for i in range(1, len(all_oof_frames)):
            merged = merged.merge(all_oof_frames[i], on=["id", "is_alert"], how="inner")
        merged.to_csv("oof_scores_merged.csv", index=False)
        print("Saved: oof_scores_merged.csv")

        rn_consensus = consensus_from_runs(all_rn_frames, mode=RN_CONSENSUS_MODE, min_votes=RN_MIN_VOTES)
        rn_consensus.to_csv(f"reliable_negatives_consensus_{RN_CONSENSUS_MODE}.csv", index=False)
        print(f"Saved: reliable_negatives_consensus_{RN_CONSENSUS_MODE}.csv | RN count: {len(rn_consensus)}")

        if CONSERVATIVE_RN and len(all_rn_cons_frames) == len(SEEDS):
            rn_consensus_c = consensus_from_runs(all_rn_cons_frames, mode=RN_CONSENSUS_MODE, min_votes=RN_MIN_VOTES)
            rn_consensus_c.to_csv(f"reliable_negatives_consensus_{RN_CONSENSUS_MODE}_conservative.csv", index=False)
            print(f"Saved: reliable_negatives_consensus_{RN_CONSENSUS_MODE}_conservative.csv | RN count: {len(rn_consensus_c)}")
    else:
        print("\nSingle-seed run. Consensus file not produced (set multiple SEEDS to enable).")
    print("\nDone.")

if __name__ == "__main__":
    main()