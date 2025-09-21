#!/usr/bin/env python3
import argparse
import os
import warnings
from collections import Counter
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------- utilities ------------------------------------
def to_int_if_numeric(col):
    try:
        return int(col)
    except (ValueError, TypeError):
        return col


def pick_feature_columns(
    df: pd.DataFrame,
    explicit_features: Iterable[int] | None,
) -> Tuple[pd.DataFrame, List[int]]:
    if not explicit_features:
        raise ValueError(
            "You must provide --features with at least one m/z integer. "
            "This script has no built-in defaults."
        )

    df = df.copy()
    df.columns = [to_int_if_numeric(c) for c in df.columns]
    exclude = {"label", "where"}

    numeric_cols = [
        c for c in df.columns if isinstance(c, (int, np.integer)) and c not in exclude
    ]
    feats = [int(f) for f in explicit_features]
    present = [mz for mz in feats if mz in numeric_cols]
    missing = [mz for mz in feats if mz not in numeric_cols]

    if not present:
        raise ValueError(
            "None of the requested --features columns are present in the CSV."
        )
    if missing:
        raise ValueError(
            f"The following requested m/z columns are missing from the CSV: {missing}. "
            f"Present features used: {present}"
        )
    return df[present], present


def _get_label_palette(labels):
    ANSI = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "fg_black": "\033[30m",
        "fg_palette": [
            "\033[36m",
            "\033[35m",
            "\033[33m",
            "\033[32m",
            "\033[34m",
            "\033[91m",
            "\033[95m",
            "\033[93m",
        ],
        "bg_palette": [
            "\033[46m",
            "\033[45m",
            "\033[43m",
            "\033[42m",
            "\033[44m",
            "\033[101m",
            "\033[105m",
            "\033[103m",
        ],
    }
    label_fg = {
        l: ANSI["fg_palette"][i % len(ANSI["fg_palette"])] for i, l in enumerate(labels)
    }
    label_bg = {
        l: ANSI["bg_palette"][i % len(ANSI["bg_palette"])] for i, l in enumerate(labels)
    }
    if "more oxidized" in label_fg:
        label_fg["more oxidized"], label_bg["more oxidized"] = ("\033[96m", "\033[106m")
    if "port" in label_fg:
        label_fg["port"], label_bg["port"] = ("\033[94m", "\033[104m")
    return ANSI, label_fg, label_bg


def _print_cm_colored(cm, labels):
    ANSI, label_fg, label_bg = _get_label_palette(labels)
    label_strs = [str(l) for l in labels]
    count_w = max(1, max(len(str(x)) for x in cm.ravel()))
    col_w = [max(len(s), count_w) for s in label_strs]
    row_w = max(len("(true)"), max(len(s) for s in label_strs))
    print("Confusion matrix")
    print(" " * (row_w + 3) + "(pred)")
    header = " | ".join(
        f'{ANSI["bold"]}{label_fg[l]}{s:^{w}}{ANSI["reset"]}'
        for l, s, w in zip(labels, label_strs, col_w)
    )
    print(f'{("(true)"):<{row_w}} | ' + header)
    for i, tl in enumerate(labels):
        left = f'{ANSI["bold"]}{label_fg[tl]}{label_strs[i]:<{row_w}}{ANSI["reset"]} | '
        cells = []
        for j, _ in enumerate(labels):
            v = cm[i, j]
            txt = f"{v:^{col_w[j]}}"
            if i == j:
                cell = f'{label_bg[tl]}{ANSI["fg_black"]}{ANSI["bold"]}{txt}{ANSI["reset"]}'
            elif v > 0:
                cell = f'{ANSI["red"]}{txt}{ANSI["reset"]}'
            else:
                cell = f'{ANSI["dim"]}{txt}{ANSI["reset"]}'
            cells.append(cell)
        print(left + " | ".join(cells))


def _print_colored_classification_report(y_true, y_pred, labels):
    ANSI, label_fg, _ = _get_label_palette(labels)
    rep = classification_report(
        y_true, y_pred, labels=labels, digits=3, output_dict=True, zero_division=0
    )
    name_w = max(len("class"), max(len(str(l)) for l in labels))
    col_w = 10
    head = (
        f'{ANSI["bold"]}{"class":<{name_w}}  '
        f'{"precision":>{col_w}}  '
        f'{"recall":>{col_w}}  '
        f'{"f1-score":>{col_w}}  '
        f'{"support":>{col_w}}{ANSI["reset"]}'
    )
    print("\nClassification report")
    print(head)
    for l in labels:
        m = rep.get(str(l), {})
        prec, rec, f1, sup = (
            m.get("precision", 0.0),
            m.get("recall", 0.0),
            m.get("f1-score", 0.0),
            int(m.get("support", 0)),
        )
        fg = label_fg[l]
        line = f"{fg}{str(l):<{name_w}}  {prec:>{col_w}.3f}  {rec:>{col_w}.3f}  {f1:>{col_w}.3f}  {sup:>{col_w}d}{ANSI['reset']}"
        print(line)
    acc = rep.get("accuracy", 0.0)
    wavg, mavg = rep.get("weighted avg", {}), rep.get("macro avg", {})
    print()
    print(f'{ANSI["bold"]}{"accuracy":<{name_w}}{ANSI["reset"]}  {acc:>{col_w}.3f}')
    print(
        f'{ANSI["bold"]}{"macro avg":<{name_w}}{ANSI["reset"]}  {mavg.get("precision",0):>{col_w}.3f}  {mavg.get("recall",0):>{col_w}.3f}  {mavg.get("f1-score",0):>{col_w}.3f}  {int(mavg.get("support",0)):>{col_w}d}'
    )
    print(
        f'{ANSI["bold"]}{"weighted avg":<{name_w}}{ANSI["reset"]}  {wavg.get("precision",0):>{col_w}.3f}  {wavg.get("recall",0):>{col_w}.3f}  {wavg.get("f1-score",0):>{col_w}.3f}  {int(wavg.get("support",0)):>{col_w}d}'
    )


# ------------------------------- training -----------------------------------
def main(
    csv_path: str,
    outdir: str,
    explicit_features: Iterable[int] | None,
    penalty: str,
    l1_ratios: List[float] | None,
    overwritemodel: bool,
):
    # ----------------- pre-check for output file -----------------
    os.makedirs(outdir, exist_ok=True)
    # We don't yet know num_features, so we'll re-load CSV minimally
    df_tmp = pd.read_csv(csv_path, nrows=5)  # just to check columns
    _, used_mz = pick_feature_columns(df_tmp, explicit_features=explicit_features)
    fname = f"model_{len(used_mz)}feat_{penalty}.joblib"
    model_out = os.path.join(outdir, fname)

    if os.path.exists(model_out) and not overwritemodel:
        resp = (
            input(f"⚠️ File '{model_out}' already exists. Overwrite? [y/N]: ")
            .strip()
            .lower()
        )
        if resp not in {"y", "yes"}:
            print("❌ Not overwriting. Exiting before training.")
            return

    # 1) Load full dataset
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Strict feature selection
    X_df, used_mz = pick_feature_columns(df, explicit_features=explicit_features)
    print(
        f"Using {len(used_mz)} m/z features:",
        used_mz[:15],
        ("..." if len(used_mz) > 15 else ""),
    )

    # Fractions (row-wise closure)
    X_raw = X_df.fillna(0.0)
    row_sums = X_raw.sum(axis=1).replace(0, np.nan)
    X_frac = X_raw.div(row_sums, axis=0).fillna(0.0)
    X = X_frac.values

    # Labels and class-check
    y = df["label"].astype(str).values
    class_counts = Counter(y)
    print("Class counts:", dict(class_counts))

    min_class = min(class_counts.values())
    if min_class < 2:
        raise SystemExit(
            "❌ Cannot run stratified cross-validation: at least one class has < 2 samples.\n"
            "   Please provide at least 2 samples per class or remove singleton classes."
        )

    # CV & model
    n_splits = min(5, min_class)
    inner_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=42)

    pipe = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            penalty=("elasticnet" if penalty == "elasticnet" else "l1"),
            solver="saga",
            class_weight="balanced",
            max_iter=50000,
            tol=1e-3,
            warm_start=True,
            random_state=42,
        ),
    )

    if penalty == "elasticnet":
        param_grid = {
            "logisticregression__C": np.logspace(-3, 2, 21),
            "logisticregression__l1_ratio": (
                l1_ratios if l1_ratios is not None else [0.7, 0.85, 0.95, 1.0]
            ),
        }
    else:  # L1
        param_grid = {
            "logisticregression__C": np.logspace(-3, 2, 21),
        }

    tuner = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    tuner.fit(X, y)

    best = tuner.best_estimator_.named_steps["logisticregression"]
    model = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            penalty=("elasticnet" if penalty == "elasticnet" else "l1"),
            solver="saga",
            C=best.C,
            l1_ratio=(best.l1_ratio if penalty == "elasticnet" else None),
            class_weight="balanced",
            max_iter=50000,
            tol=1e-3,
            warm_start=True,
            random_state=42,
        ),
    ).fit(X, y)

    diag_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    y_pred = cross_val_predict(model, X, y, cv=diag_cv, method="predict")

    print("CV Accuracy (single partition):", accuracy_score(y, y_pred))
    labels = sorted(class_counts.keys())
    cm = confusion_matrix(y, y_pred, labels=labels)
    _print_cm_colored(cm, labels)
    _print_colored_classification_report(y, y_pred, labels)

    # 4) Persist
    try:
        import joblib

        joblib.dump({"model": model, "used_mz": used_mz, "penalty": penalty}, model_out)
        print(f"💾 Saved model (with feature list) to {model_out}")
    except Exception as e:
        print("Note: could not save model:", e)


# ---------------------------------- CLI -------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with spectra and labels")
    ap.add_argument(
        "--outdir",
        required=True,
        help="Folder to store the generated model file (will be created if missing)",
    )
    ap.add_argument(
        "--features",
        nargs="+",
        type=int,
        required=True,
        help="Explicit list of m/z integers to use. No defaults; no fallbacks.",
    )
    ap.add_argument(
        "--penalty",
        choices=["l1", "elasticnet"],
        required=True,
        help="Sparse regularization: 'l1' (lasso) or 'elasticnet'.",
    )
    ap.add_argument(
        "--l1-ratios",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Elastic-Net l1_ratio grid (required if --penalty elasticnet). "
            "l1_ratio=1.0 is pure L1 (sparse feature selection). "
            "Smaller values (e.g. 0.5–0.9) add stability with correlated features. "
            "Example: --l1-ratios 0.5 0.75 0.9 1.0"
        ),
    )
    ap.add_argument(
        "--quiet", action="store_true", help="Suppress scikit-learn ConvergenceWarnings"
    )
    ap.add_argument(
        "--overwritemodel",
        action="store_true",
        help="Overwrite existing model file without asking",
    )
    args = ap.parse_args()

    # Validation
    if args.penalty == "l1":
        if args.l1_ratios is not None:
            ap.error("--l1-ratios is only valid when --penalty elasticnet")
    else:  # penalty == "elasticnet"
        if args.l1_ratios is None:
            ap.error(
                "--l1-ratios is required when --penalty elasticnet "
                "(e.g. --l1-ratios 0.5 0.75 0.9 1.0)"
            )

    if args.quiet:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    main(
        args.csv,
        args.outdir,
        explicit_features=args.features,
        penalty=args.penalty,
        l1_ratios=args.l1_ratios if args.penalty == "elasticnet" else None,
        overwritemodel=args.overwritemodel,
    )
