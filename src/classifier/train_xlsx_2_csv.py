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
from decimal import Decimal

import numpy as np
import pandas as pd


def convert_excel_to_csv(
    input_path: str, output_path: str, sheet_name: str = "Sheet1"
) -> None:
    # Read raw with no header so we can detect header/label rows robustly
    raw = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    if raw.empty:
        raise ValueError(f"Sheet '{sheet_name}' appears to be empty.")

    # Find the header row by locating the row whose first non-empty cell equals 'm/z' (case-insensitive)
    col0 = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    # Allow variations like 'm/z', 'm-z', 'mz'
    header_row_candidates = col0[col0.isin({"m/z", "m-z", "mz"})].index.tolist()
    if not header_row_candidates:
        # Fallback: search anywhere for a header cell equal to m/z
        mask_any = raw.applymap(
            lambda x: str(x).strip().lower() in {"m/z", "m-z", "mz"}
        ).any(axis=1)
        header_row_candidates = raw.index[mask_any].tolist()
    if not header_row_candidates:
        raise ValueError("Could not locate the header row (cell 'm/z').")

    header_row_idx = header_row_candidates[0]

    # The row above the header is treated as the "label" row (e.g., HOA). If missing, we leave label blank.
    label_row = (
        raw.iloc[header_row_idx - 1]
        if header_row_idx > 0
        else pd.Series([np.nan] * raw.shape[1])
    )

    # Build column names from the detected header row
    headers = raw.iloc[header_row_idx].tolist()
    headers[0] = "m/z"  # ensure exact label
    data = raw.iloc[header_row_idx + 1 :].copy()
    data.columns = headers

    # Drop columns with entirely empty header names (e.g., trailing empty columns)
    data = data.loc[
        :,
        [
            not (isinstance(c, float) and pd.isna(c)) and str(c).strip() != ""
            for c in data.columns
        ],
    ]

    # Coerce m/z to numeric and drop non-numeric/blank rows
    data["m/z"] = pd.to_numeric(data["m/z"], errors="coerce")
    data = data.dropna(subset=["m/z"]).sort_values("m/z")

    # Identify profile columns
    profile_cols = [c for c in data.columns if c != "m/z"]
    if not profile_cols:
        raise ValueError("No profile columns found besides 'm/z'.")

    # Map profile -> label using the "label" row aligned by column position
    label_map = {}
    for j, name in enumerate(headers):
        if j == 0:
            continue
        if (
            name is None
            or (isinstance(name, float) and pd.isna(name))
            or str(name).strip() == ""
        ):
            continue
        tval = label_row.iloc[j] if j < len(label_row) else np.nan
        tval = "" if pd.isna(tval) else str(tval).strip()
        label_map[str(name)] = tval

    # Create wide matrix indexed by m/z where each column is a profile
    wide = data.set_index("m/z")[profile_cols]
    if wide.index.duplicated().any():
        wide = wide.groupby(level=0).first()

    # Transpose: rows -> profiles, columns -> m/z
    out = wide.T

    # Ensure column order is ascending by numeric m/z
    try:
        mz_sorted = sorted(out.columns, key=float)
    except Exception:
        mz_sorted = list(out.columns)
    out = out.reindex(columns=mz_sorted)

    # Add the 'where' and 'label' columns
    out.insert(0, "label", out.index.map(lambda prof: label_map.get(str(prof), "")))
    out.insert(0, "where", out.index.astype(str))

    # ---- Write CSV with desired quoting rules ----
    # Header: unquoted; Data: strings quoted, numerics unquoted; NaN as unquoted NaN
    def _fmt_col_name(c):
        # Make m/z headers like 12.0 appear as 12
        try:
            f = float(c)
            return str(int(f)) if f.is_integer() else str(c)
        except Exception:
            return str(c)

    header_cols = [_fmt_col_name(c) for c in out.columns]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # write header manually (unquoted)
        f.write(",".join(header_cols) + "\n")

        writer = csv.writer(
            f,
            quoting=csv.QUOTE_NONNUMERIC,
            lineterminator="\n",  # <- force Unix line endings, no ^M
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
