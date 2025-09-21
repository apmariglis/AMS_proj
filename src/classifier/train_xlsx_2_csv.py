#!/usr/bin/env python3
"""
xlsx_to_profile_csv.py

Read "Sheet1" (or a chosen sheet) from an Excel workbook where the layout is:
- A "label" row (e.g., 'HOA', 'COA', etc.) above the real header row
- A header row containing "m/z" in the first column and profile names in subsequent columns
- Data rows with m/z values and corresponding profile intensities

Transform it into a CSV where each profile becomes a row and m/z values become columns.

Usage:
    python xlsx_to_profile_csv.py --input input.xlsx --output output.csv [--sheet SHEETNAME]

If --sheet is omitted, "Sheet1" is used.
"""
import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd


def _make_unique(names):
    """
    Return a list of names where duplicates are given stable suffixes
    like 'Name [#1]', 'Name [#2]' in left-to-right order.
    """
    counts = Counter(names)
    seen = defaultdict(int)
    out = []
    for nm in names:
        if counts[nm] <= 1:
            out.append(str(nm))
        else:
            seen[nm] += 1
            out.append(f"{nm} [#{seen[nm]}]")
    return out


# --- snip: imports and helpers stay the same (including _make_unique) ---


def convert_excel_to_csv(
    input_path: str, output_path: str, sheet_name: str = "Sheet1"
) -> None:
    raw = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    if raw.empty:
        raise ValueError(f"Sheet '{sheet_name}' appears to be empty.")

    # Find header row
    col0 = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    header_row_candidates = col0[col0.isin({"m/z", "m-z", "mz"})].index.tolist()
    if not header_row_candidates:
        mask_any = raw.applymap(
            lambda x: str(x).strip().lower() in {"m/z", "m-z", "mz"}
        ).any(axis=1)
        header_row_candidates = raw.index[mask_any].tolist()
    if not header_row_candidates:
        raise ValueError("Could not locate the header row (cell 'm/z').")
    header_row_idx = header_row_candidates[0]

    # Label row and header row (raw, unfiltered)
    label_row = (
        raw.iloc[header_row_idx - 1]
        if header_row_idx > 0
        else pd.Series([np.nan] * raw.shape[1])
    )
    headers = raw.iloc[header_row_idx].tolist()
    headers[0] = "m/z"

    # Build a keep mask ONCE and reuse for everything (headers, label_row, data)
    def _keep(col_name):
        return (
            not (isinstance(col_name, float) and pd.isna(col_name))
            and str(col_name).strip() != ""
        )

    keep_mask = [_keep(c) for c in headers]

    # Filter headers and label_row with the SAME mask so positions align
    headers_f = [h for h, k in zip(headers, keep_mask) if k]
    label_row_f = pd.Series([v for v, k in zip(label_row.tolist(), keep_mask) if k])

    # Slice the data block and apply the same column mask
    data = raw.iloc[header_row_idx + 1 :, :].copy()
    data.columns = headers
    data = data.loc[:, keep_mask]

    # Coerce m/z and clean rows
    if "m/z" not in data.columns:
        raise ValueError("After filtering, 'm/z' column is missing.")
    data["m/z"] = pd.to_numeric(data["m/z"], errors="coerce")
    data = data.dropna(subset=["m/z"]).sort_values("m/z")

    # Identify profile columns from the FILTERED headers
    profile_cols = [c for c in headers_f if c != "m/z"]
    if not profile_cols:
        raise ValueError("No profile columns found besides 'm/z'.")

    # Create labels by POSITION from the FILTERED label row (skip first 'm/z' entry)
    labels = []
    for v in label_row_f.tolist()[1:]:
        labels.append("" if pd.isna(v) else str(v).strip())

    # Disambiguate duplicate profile names (position-based)
    where_unique = _make_unique([str(c) for c in profile_cols])

    # Create wide matrix indexed by m/z where each column is a profile
    # IMPORTANT: after filtering, data_f has "m/z" + the profile columns.
    data_f = data  # already filtered by keep_mask
    mz_col = "m/z"
    if mz_col not in data_f.columns:
        raise ValueError("After filtering, 'm/z' column is missing.")

    # Build wide WITHOUT label-based selection (avoids duplicate-name expansion)
    # and WITHOUT positional iloc (avoids shifting after set_index).
    wide = data_f.set_index(mz_col)  # leaves only profile columns as columns

    if wide.index.duplicated().any():
        wide = wide.groupby(level=0).first()

    # Transpose: rows -> profiles, columns -> m/z
    out = wide.T

    # Ensure m/z columns are sorted numerically
    try:
        mz_sorted = sorted(out.columns, key=float)
    except Exception:
        mz_sorted = list(out.columns)
    out = out.reindex(columns=mz_sorted)

    # --- Robust length check ---
    n_profiles = out.shape[0]
    if not (len(labels) == len(where_unique) == n_profiles):
        raise ValueError(
            f"Alignment error: profiles={n_profiles}, labels={len(labels)}, "
            f"where={len(where_unique)}"
        )

    # Inject metadata columns expected by the writer below
    out.insert(0, "label", labels)  # add labels column first
    out.insert(0, "where", where_unique)  # then where column
    out.index = range(len(out))  # clear the old index

    # Writer (unchanged)
    def _fmt_col_name(c):
        try:
            f = float(c)
            return str(int(f)) if f.is_integer() else str(c)
        except Exception:
            return str(c)

    header_cols = [_fmt_col_name(c) for c in out.columns]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header_cols) + "\n")
        writer = csv.writer(
            f,
            quoting=csv.QUOTE_NONNUMERIC,
            lineterminator="\n",
        )
        for row in out.itertuples(index=False, name=None):
            where_val, label_val, *vals = row
            out_row = [str(where_val), str(label_val)]
            for v in vals:
                out_row.append(Decimal("NaN") if pd.isna(v) else float(v))
            writer.writerow(out_row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to the input .xlsx")
    parser.add_argument("--output", help="Path to the output .csv")
    parser.add_argument("--sheet", help="Worksheet name (default: Sheet1)")
    args = parser.parse_args()
    convert_excel_to_csv(args.input, args.output, args.sheet)


if __name__ == "__main__":
    main()
