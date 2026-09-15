#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a true nnU-Net v2 model on all cases in case_table.csv.

The script reuses the nnU-Net raw dataset created by train_nnUNet_v2.py,
runs nnUNetv2_predict, computes Dice per case, writes data_splits.xlsx with
train/val/test sheets, and writes dice_summary.csv with split summaries.
"""

import argparse
import datetime
import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import params
from tools.datasetgenerator import (
    datasetgenerator_bergen,
    datasetgenerator_hong_kong,
    datasetgenerator_mont,
)
from tools.utils import (
    build_aligned_image_path_nnUNet,
    build_aligned_mask_path_nnUNet,
    copy_or_link_file_nnUNet,
    get_nnunet_env_nnUNet,
    read_hf_token_file,
    require_nnunet_cli_nnUNet,
    resolve_model_dir_from_hf,
    run_command_nnUNet,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate true nnU-Net v2 predictions for all cases in case_table.csv."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Wrapper model directory, Hugging Face repo id, or huggingface.co URL.",
    )
    parser.add_argument("--hf_revision", default=None)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_token_file", default=None)
    parser.add_argument(
        "--case_table",
        default="",
        help=(
            "Optional CSV/Excel case table. Must contain at least subj. "
            "Multi-channel imgpath uses semicolon-separated paths in model modality order."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["bergen", "mont", "hong_kong"],
        help=(
            "Project datasets to evaluate, using params.py and datasetgenerator.py. "
            "Mutually exclusive with --case_table."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pth",
        help="nnU-Net checkpoint to use, for example checkpoint_best.pth or checkpoint_final.pth.",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Alias for --output_dir.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional output directory. Defaults to prediction/ under the local model folder.",
    )
    parser.add_argument(
        "--output_prefix",
        default="",
        help="Optional filename prefix. Defaults to this script name.",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument(
        "--file_mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to create the temporary nnU-Net prediction input files.",
    )
    parser.add_argument(
        "--gpu",
        default="",
        help="Optional GPU id. Sets CUDA_VISIBLE_DEVICES for nnUNetv2_predict.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--skip_predict", action="store_true")
    parser.add_argument("--npp", type=int, default=3)
    parser.add_argument("--nps", type=int, default=3)
    return parser.parse_args()


def load_metadata(model_dir):
    settings_path = os.path.join(model_dir, "settings.json")
    if not os.path.isfile(settings_path):
        raise FileNotFoundError(f"Fant ikke settings.json: {settings_path}")
    with open(settings_path) as f:
        return json.load(f)


def safe_folder_name(value):
    value = str(value).strip().replace("hf://", "")
    value = value.replace("/", "__").replace("\\", "__")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def resolve_output_model_dir(input_model_ref, resolved_model_dir, meta):
    if os.path.exists(os.path.expanduser(input_model_ref)):
        return resolved_model_dir

    model_name = meta.get("model_name") or os.path.basename(input_model_ref.rstrip("/"))
    local_model_dir = os.path.join(params.prepathmodels, safe_folder_name(model_name))
    os.makedirs(local_model_dir, exist_ok=True)

    src_settings = os.path.join(resolved_model_dir, "settings.json")
    dst_settings = os.path.join(local_model_dir, "settings.json")
    if os.path.isfile(src_settings) and not os.path.exists(dst_settings):
        shutil.copy2(src_settings, dst_settings)

    return local_model_dir


def localize_nnunet_paths(meta, model_dir):
    meta = dict(meta)
    base_dir = meta.get("nnunet_base_dir", "")
    if base_dir and os.path.isdir(base_dir):
        return meta

    candidate = os.path.join(model_dir, "nnUNet_work")
    if not os.path.isdir(candidate):
        return meta

    dataset_name = meta["nnunet_dataset_name"]
    meta["nnunet_base_dir"] = candidate
    meta["nnunet_raw"] = os.path.join(candidate, "nnUNet_raw")
    meta["nnunet_preprocessed"] = os.path.join(candidate, "nnUNet_preprocessed")
    meta["nnunet_results"] = os.path.join(candidate, "nnUNet_results")
    meta["nnunet_raw_dataset_dir"] = os.path.join(
        meta["nnunet_raw"],
        dataset_name,
    )
    return meta


def nnunet_base_dir_from_meta(meta):
    if meta.get("nnunet_base_dir"):
        return meta["nnunet_base_dir"]
    if meta.get("nnunet_raw"):
        return os.path.dirname(os.path.abspath(meta["nnunet_raw"]))
    raise KeyError(
        "settings.json mangler både 'nnunet_base_dir' og 'nnunet_raw'. "
        "Kan ikke sette nnU-Net-miljøvariabler."
    )


def checkpoint_tag(checkpoint):
    tag = os.path.splitext(os.path.basename(checkpoint))[0]
    return tag.replace("checkpoint_", "")


def raw_image_path(raw_dataset_dir, split, case_id, channel_idx):
    image_dir = "imagesTs" if split == "test" else "imagesTr"
    return os.path.join(
        raw_dataset_dir,
        image_dir,
        f"{case_id}_{channel_idx:04d}.nii.gz",
    )


def raw_label_path(raw_dataset_dir, split, case_id):
    label_dir = "labelsTs" if split == "test" else "labelsTr"
    return os.path.join(raw_dataset_dir, label_dir, f"{case_id}.nii.gz")


def prepare_prediction_input(case_table, meta, input_dir, file_mode):
    raw_dataset_dir = meta["nnunet_raw_dataset_dir"]
    n_channels = int(meta["num_channels"])
    os.makedirs(input_dir, exist_ok=True)

    for _, row in case_table.iterrows():
        case_id = row["case_id"]
        split = row["split"]
        fallback_paths = str(row.get("imgpath", "")).split(";")

        for channel_idx in range(n_channels):
            src = raw_image_path(raw_dataset_dir, split, case_id, channel_idx)
            if not os.path.isfile(src) and channel_idx < len(fallback_paths):
                src = fallback_paths[channel_idx]
            if not os.path.isfile(src):
                raise FileNotFoundError(
                    f"Mangler input for {case_id}, kanal {channel_idx}: {src}"
                )

            dst = os.path.join(input_dir, f"{case_id}_{channel_idx:04d}.nii.gz")
            copy_or_link_file_nnUNet(src, dst, mode=file_mode)


def prediction_path(output_dir, case_id):
    return os.path.join(output_dir, f"{case_id}.nii.gz")


def label_path(case_row, meta):
    raw_path = raw_label_path(
        meta["nnunet_raw_dataset_dir"],
        case_row["split"],
        case_row["case_id"],
    )
    if os.path.isfile(raw_path):
        return raw_path

    fallback = case_row.get("pathmask", "")
    if os.path.isfile(fallback):
        return fallback

    raise FileNotFoundError(
        f"Mangler label for {case_row['case_id']}: {raw_path} / {fallback}"
    )


def dice_from_files(pred_path, label_path_value):
    pred = nib.load(pred_path).get_fdata() > 0
    label = nib.load(label_path_value).get_fdata() > 0

    if pred.shape != label.shape:
        raise ValueError(
            f"Prediksjon og label har ulik shape for {pred_path}: "
            f"{pred.shape} vs {label.shape}"
        )

    intersection = np.logical_and(pred, label).sum(dtype=np.float64)
    denominator = pred.sum(dtype=np.float64) + label.sum(dtype=np.float64)
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def read_case_table(path, split="all"):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        split_sheets = [s for s in ["train", "val", "test"] if s in xls.sheet_names]
        if split_sheets:
            sheets = split_sheets if split == "all" else [split]
            missing = [s for s in sheets if s not in xls.sheet_names]
            if missing:
                raise ValueError(f"Mangler sheet(s) i {path}: {missing}")
            parts = []
            for sheet in sheets:
                df_sheet = pd.read_excel(path, sheet_name=sheet)
                if "split" not in df_sheet.columns:
                    df_sheet["split"] = sheet
                parts.append(df_sheet)
            return pd.concat(parts, ignore_index=True)

        sheet_name = "cases" if "cases" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet_name)

    return pd.read_csv(path, sep=None, engine="python")


def output_sheet_name(path, split):
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".xlsx", ".xls"}:
        return "cases"

    xls = pd.ExcelFile(path)
    split_sheets = [s for s in ["train", "val", "test"] if s in xls.sheet_names]
    if split != "all" and split in xls.sheet_names:
        return split
    if split_sheets and split == "all":
        return "cases"
    if "cases" in xls.sheet_names:
        return "cases"
    return xls.sheet_names[0]


def default_case_table_path(model_dir, meta, output_model_dir=None):
    candidates = [
        os.path.join(model_dir, "case_table.xlsx"),
        os.path.join(model_dir, meta.get("case_table_file", "case_table.csv")),
        os.path.join(model_dir, "case_table.csv"),
        os.path.join(model_dir, "data_splits.xlsx"),
    ]
    if output_model_dir and output_model_dir != model_dir:
        candidates.extend(
            [
                os.path.join(output_model_dir, "case_table.xlsx"),
                os.path.join(output_model_dir, meta.get("case_table_file", "case_table.csv")),
                os.path.join(output_model_dir, "case_table.csv"),
                os.path.join(output_model_dir, "data_splits.xlsx"),
            ]
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def build_project_case_table(datasets, modalities, reference):
    dfs = []
    for dataset in datasets:
        if dataset not in params.DATASET_CONFIG:
            raise ValueError(f"Ukjent datasett: {dataset}")
        available = params.DATASET_CONFIG[dataset]["modalities"].keys()
        for modality in modalities:
            if modality not in available:
                raise ValueError(
                    f"Modalitet '{modality}' finnes ikke i datasett '{dataset}'. "
                    f"Tilgjengelig: {list(available)}"
                )

        mod_info = params.DATASET_CONFIG[dataset]["modalities"]
        require = [mod_info[m]["col"] for m in modalities]
        list_path = getattr(params, f"pathlistvalid_{dataset}")

        df_source = pd.read_csv(list_path, sep=";").groupby(
            "subj", as_index=False
        ).first()
        if dataset == "bergen":
            df = datasetgenerator_bergen(df_source, require)
        elif dataset == "mont":
            df = datasetgenerator_mont(df_source, require)
        elif dataset == "hong_kong":
            df = datasetgenerator_hong_kong(df_source, require)
        else:
            raise ValueError(f"Ukjent datasett: {dataset}")

        df["source"] = dataset
        dfs.append(df)

    if not dfs:
        raise ValueError("Ingen datasett valgt")

    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[df.dataset == "man"].reset_index(drop=True)
    df["imgpath"] = [
        build_aligned_image_path_nnUNet(
            subj=subj,
            modalities=modalities,
            prepathnii=getattr(params, f"prepathnifti_{source}"),
            reference=reference,
            dataset=source,
        )
        for subj, source in zip(df.subj, df.source)
    ]
    df["pathmask"] = [
        build_aligned_mask_path_nnUNet(
            subj=subj,
            maskname=maskname,
            prepathnii=getattr(params, f"prepathnifti_{source}"),
            reference=reference,
            dataset=source,
        )
        for subj, maskname, source in zip(df.subj, df.pathmask, df.source)
    ]
    return df


def complete_case_table(case_table, case_table_path, model_dir, output_model_dir, meta, split):
    if "subj" not in case_table.columns:
        raise ValueError(f"Missing required column in {case_table_path}: ['subj']")

    case_table = case_table.copy()
    if "split" not in case_table.columns:
        case_table["split"] = "test"
    if "case_id" not in case_table.columns:
        case_table["case_id"] = [
            f"pred_{idx:04d}_{safe_folder_name(subj)}"
            for idx, subj in enumerate(case_table["subj"].astype(str).values)
        ]

    required_for_eval = {"imgpath", "pathmask"}
    missing_for_eval = required_for_eval.difference(case_table.columns)
    if not missing_for_eval:
        return case_table

    lookup_path = default_case_table_path(model_dir, meta, output_model_dir=output_model_dir)
    if not os.path.isfile(lookup_path):
        raise ValueError(
            f"Missing columns needed for nnU-Net evaluation in {case_table_path}: "
            f"{sorted(missing_for_eval)}. Could not find local model "
            "case_table.xlsx/csv to look them up by subj. For HF models, pass "
            "imgpath and pathmask in --case_table."
        )

    lookup = read_case_table(lookup_path, split=split)
    if "subj" not in lookup.columns:
        raise ValueError(f"Lookup table is missing subj column: {lookup_path}")
    missing_lookup = required_for_eval.difference(lookup.columns)
    if missing_lookup:
        raise ValueError(
            f"Lookup table {lookup_path} is missing required columns: "
            f"{sorted(missing_lookup)}"
        )

    lookup_cols = ["subj"] + [c for c in required_for_eval if c not in case_table.columns]
    lookup = lookup[lookup_cols].drop_duplicates("subj")
    merged = case_table.merge(lookup, on="subj", how="left")

    still_missing = [
        c for c in required_for_eval
        if c not in merged.columns or merged[c].isna().any()
    ]
    if still_missing:
        missing_subj = merged.loc[
            merged[still_missing[0]].isna(), "subj"
        ].astype(str).head(10).tolist()
        raise ValueError(
            f"Could not fill columns {still_missing} for all subjects. "
            f"First missing subjects: {missing_subj}"
        )

    return merged


def add_dice_scores(case_table, meta, output_dir):
    rows = []
    for _, row in case_table.iterrows():
        row = row.copy()
        pred_path = prediction_path(output_dir, row["case_id"])
        gt_path = label_path(row, meta)

        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"Mangler prediksjon: {pred_path}")

        row["prediction_path"] = pred_path
        row["label_path"] = gt_path
        row["dice"] = dice_from_files(pred_path, gt_path)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_dice(case_table):
    rows = []
    for split in ("train", "val", "test"):
        values = case_table.loc[case_table["split"] == split, "dice"].astype(float)
        if len(values) == 0:
            continue

        q1, median, q3 = np.percentile(values, [25, 50, 75])
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        rows.append(
            {
                "split": split,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(std),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "mean_std": f"{np.mean(values):.3f} ({std:.3f})",
                "median_iqr": f"{median:.3f} ({q1:.3f}-{q3:.3f})",
            }
        )
    return pd.DataFrame(rows)


def dice_summary_stats(values):
    values = pd.Series(values).dropna().astype(float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean_std": f"{np.mean(values):.4f} ({std:.4f})",
        "median_iqr": f"{median:.4f} ({q1:.4f}-{q3:.4f})",
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def write_text_summary(path, scored, model_ref, model_dir, dataset_path, checkpoint):
    total = dice_summary_stats(scored["dice"])
    with open(path, "w") as f:
        f.write("Dice summary\n")
        f.write("============\n")
        f.write(f"Model ref   : {model_ref}\n")
        f.write(f"Model dir   : {model_dir}\n")
        f.write(f"Dataset     : {dataset_path}\n")
        f.write(f"Checkpoint  : {checkpoint}\n")
        f.write(f"n           : {total['n']}\n")
        f.write(f"Mean (std)  : {total['mean_std']}\n")
        f.write(f"Median (IQR): {total['median_iqr']}\n")
        f.write(f"Min-max     : {total['min']:.4f}-{total['max']:.4f}\n")

        if "split" in scored.columns:
            f.write("\nDice by split\n")
            f.write("=============\n")
            for split_value, group in scored.groupby("split", dropna=False):
                stats = dice_summary_stats(group["dice"])
                f.write(f"\nSplit       : {split_value}\n")
                f.write(f"n           : {stats['n']}\n")
                f.write(f"Mean (std)  : {stats['mean_std']}\n")
                f.write(f"Median (IQR): {stats['median_iqr']}\n")
                f.write(f"Min-max     : {stats['min']:.4f}-{stats['max']:.4f}\n")


def main():
    args = parse_args()
    if args.datasets and args.case_table:
        raise ValueError("Use either --datasets or --case_table, not both.")

    input_model_ref = args.model_dir
    hf_token = args.hf_token or read_hf_token_file(args.hf_token_file)
    model_dir = resolve_model_dir_from_hf(
        args.model_dir,
        revision=args.hf_revision,
        cache_dir=args.hf_cache_dir,
        token=hf_token,
    )
    meta = localize_nnunet_paths(load_metadata(model_dir), model_dir)
    output_model_dir = resolve_output_model_dir(input_model_ref, model_dir, meta)

    if args.datasets:
        case_table_path = "datasets=" + ",".join(args.datasets)
        case_table = build_project_case_table(
            datasets=args.datasets,
            modalities=list(meta["modalities"]),
            reference=meta.get("reference", meta["modalities"][0]),
        )
    else:
        case_table_path = (
            args.case_table
            or default_case_table_path(model_dir, meta, output_model_dir=output_model_dir)
        )
        if not os.path.isfile(case_table_path):
            raise FileNotFoundError(f"Fant ikke case table: {case_table_path}")

        case_table = read_case_table(case_table_path, split=args.split)
        if args.split != "all" and "split" in case_table.columns:
            case_table = case_table[case_table["split"].astype(str) == args.split].copy()
            if case_table.empty:
                raise ValueError(f"No rows with split='{args.split}' in {case_table_path}")
        case_table = complete_case_table(
            case_table=case_table,
            case_table_path=case_table_path,
            model_dir=model_dir,
            output_model_dir=output_model_dir,
            meta=meta,
            split=args.split,
        )
    tag = checkpoint_tag(args.checkpoint)
    output_dir = (
        os.path.abspath(args.out_dir or args.output_dir)
        if (args.out_dir or args.output_dir)
        else os.path.join(output_model_dir, "prediction")
    )
    input_dir = os.path.join(output_dir, "input")
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    prepare_prediction_input(
        case_table=case_table,
        meta=meta,
        input_dir=input_dir,
        file_mode=args.file_mode,
    )

    if not args.skip_predict:
        require_nnunet_cli_nnUNet()
        env, _ = get_nnunet_env_nnUNet(nnunet_base_dir_from_meta(meta))
        if args.gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        if os.path.isdir(prediction_dir):
            shutil.rmtree(prediction_dir)

        cmd = [
            "nnUNetv2_predict",
            "-i",
            input_dir,
            "-o",
            prediction_dir,
            "-d",
            str(meta["nnunet_dataset_id"]),
            "-c",
            meta["nnunet_configuration"],
            "-f",
            str(meta["nnunet_fold"]),
            "-tr",
            meta["nnunet_trainer"],
            "-p",
            meta["nnunet_plans"],
            "-chk",
            args.checkpoint,
            "-device",
            args.device,
            "-npp",
            str(args.npp),
            "-nps",
            str(args.nps),
        ]
        if args.disable_tta:
            cmd.append("--disable_tta")

        run_command_nnUNet(cmd, env=env)

    scored = add_dice_scores(case_table, meta, prediction_dir)
    summary = summarize_dice(scored)

    run_id = datetime.datetime.now().strftime("%Y%m%d")
    output_prefix = args.output_prefix or os.path.splitext(os.path.basename(__file__))[0]
    excel_path = os.path.join(output_dir, f"{output_prefix}_results_{run_id}.xlsx")
    summary_txt = os.path.join(output_dir, f"{output_prefix}_summary_{run_id}.txt")

    sheet_name = output_sheet_name(case_table_path, args.split)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        scored.to_excel(writer, sheet_name=sheet_name, index=False)
    write_text_summary(
        summary_txt,
        scored=scored,
        model_ref=args.model_dir,
        model_dir=output_model_dir,
        dataset_path=case_table_path,
        checkpoint=args.checkpoint,
    )

    print(f"Excel med Dice skrevet til: {excel_path}")
    print(f"Dice-oppsummering skrevet til: {summary_txt}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
