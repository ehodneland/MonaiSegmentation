#!/usr/bin/env python3
import argparse
import os

import pandas as pd


SPLIT_FILES = {
    "train": "dftrain.csv",
    "val": "dfval.csv",
    "test": "dftest.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create standardized case_table.xlsx/csv files in model directories."
    )
    parser.add_argument(
        "--models_dir",
        default="/raid/erlend/Dropbox/Precision_Imaging_in_Gynecologic_Cancer/EC/MonaiSegmentation/models",
        help="Directory containing model subdirectories.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be changed without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing case_table.xlsx/csv files.",
    )
    parser.add_argument(
        "--include_nested",
        action="store_true",
        help="Also scan nested model directories, for example ensemble fold folders.",
    )
    return parser.parse_args()


def read_csv_auto(path):
    return pd.read_csv(path, sep=None, engine="python")


def add_split(df, split):
    out = df.copy()
    out["split"] = split
    return out


def frontload_columns(df):
    front = [
        col for col in ["case_id", "subj", "source", "dataset", "split", "imgpath", "pathmask"]
        if col in df.columns
    ]
    rest = [col for col in df.columns if col not in front]
    return df[front + rest]


def case_table_from_xlsx(model_dir):
    path = os.path.join(model_dir, "data_splits.xlsx")
    if not os.path.isfile(path):
        return None, None

    xl = pd.ExcelFile(path)
    parts = []
    for split in ["train", "val", "test"]:
        if split in xl.sheet_names:
            parts.append(add_split(pd.read_excel(path, sheet_name=split), split))

    if not parts:
        return None, None
    return frontload_columns(pd.concat(parts, ignore_index=True)), "data_splits.xlsx"


def case_table_from_split_csvs(model_dir):
    parts = []
    used = []
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            parts.append(add_split(read_csv_auto(path), split))
            used.append(filename)

    if not parts:
        return None, None
    return frontload_columns(pd.concat(parts, ignore_index=True)), ", ".join(used)


def case_table_from_existing_csv(model_dir):
    path = os.path.join(model_dir, "case_table.csv")
    if not os.path.isfile(path):
        return None, None

    df = read_csv_auto(path)
    if "split" not in df.columns:
        return None, None
    return frontload_columns(df), "case_table.csv"


def build_case_table(model_dir):
    for builder in [
        case_table_from_existing_csv,
        case_table_from_xlsx,
        case_table_from_split_csvs,
    ]:
        table, source = builder(model_dir)
        if table is not None:
            return table, source
    return None, None


def iter_model_dirs(models_dir, include_nested=False):
    max_depth = None if include_nested else 1
    models_dir = os.path.abspath(os.path.expanduser(models_dir))
    for root, dirs, _ in os.walk(models_dir):
        rel = os.path.relpath(root, models_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth == 0:
            continue
        if max_depth is not None and depth > max_depth:
            dirs[:] = []
            continue
        yield root


def write_case_table(model_dir, table):
    xlsx_path = os.path.join(model_dir, "case_table.xlsx")
    csv_path = os.path.join(model_dir, "case_table.csv")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="cases", index=False)
    table.to_csv(csv_path, index=False, sep=";")
    return xlsx_path, csv_path


def main():
    args = parse_args()
    models_dir = os.path.abspath(os.path.expanduser(args.models_dir))
    if not os.path.isdir(models_dir):
        raise NotADirectoryError(f"--models_dir is not a directory: {models_dir}")

    changed = 0
    skipped = 0
    missing = 0
    errors = []

    for model_dir in iter_model_dirs(models_dir, include_nested=args.include_nested):
        existing_xlsx = os.path.exists(os.path.join(model_dir, "case_table.xlsx"))
        existing_csv = os.path.exists(os.path.join(model_dir, "case_table.csv"))
        if (existing_xlsx or existing_csv) and not args.overwrite:
            skipped += 1
            print(f"SKIP existing: {model_dir}")
            continue

        try:
            table, source = build_case_table(model_dir)
            if table is None:
                missing += 1
                continue
            if "split" not in table.columns:
                raise ValueError("case table has no split column")
            invalid = sorted(set(table["split"].dropna()) - {"train", "val", "test"})
            if invalid:
                raise ValueError(f"unexpected split values: {invalid}")

            counts = table["split"].value_counts().to_dict()
            if args.dry_run:
                print(f"DRY {model_dir}: source={source}, rows={len(table)}, counts={counts}")
            else:
                xlsx_path, csv_path = write_case_table(model_dir, table)
                print(f"WROTE {model_dir}: source={source}, rows={len(table)}, counts={counts}")
                print(f"  - {xlsx_path}")
                print(f"  - {csv_path}")
            changed += 1
        except Exception as exc:
            errors.append((model_dir, str(exc)))
            print(f"ERROR {model_dir}: {exc}")

    print("\nSummary")
    print(f"  candidates written/planned : {changed}")
    print(f"  skipped existing           : {skipped}")
    print(f"  no split files found       : {missing}")
    print(f"  errors                     : {len(errors)}")
    if errors:
        for model_dir, error in errors[:20]:
            print(f"  - {model_dir}: {error}")


if __name__ == "__main__":
    main()
