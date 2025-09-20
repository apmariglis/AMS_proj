import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export Excel sheet to CSV.")
    parser.add_argument("--input", help="Path to the input Excel file")
    parser.add_argument("--output", help="Path to the output CSV file")
    parser.add_argument("--sheet", help="Name of the sheet to export")

    args = parser.parse_args()

    # Read the given sheet and save as CSV
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    df.to_csv(args.output, index=False)

    print(f'Sheet "{args.sheet}" has been exported to {args.output}')


if __name__ == "__main__":
    main()
