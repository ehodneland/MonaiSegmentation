#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a true nnU-Net v2 model on all cases in case_table.csv.

The script reuses the nnU-Net raw dataset created by train_nnUNet_v2.py,
runs nnUNetv2_predict, computes Dice per case, writes data_splits.xlsx with
train/val/test sheets, and writes dice_summary.csv with split summaries.
"""

import argparse
import json
import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd

from utils import (
    copy_or_link_file_nnUNet,
    get_nnunet_env_nnUNet,
    require_nnunet_cli_nnUNet,
    run_command_nnUNet,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate true nnU-Net v2 predictions for all cases in case_table.csv."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Wrapper model directory containing settings.json and case_table.csv.",
    )
    parser.add_argument(
        "--case_table",
        default="",
        help="Optional path to case_table.csv. Defaults to <model_dir>/case_table.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pth",
        help="nnU-Net checkpoint to use, for example checkpoint_best.pth or checkpoint_final.pth.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional output directory. Defaults to <model_dir>/nnUNet_eval_<checkpoint>.",
    )
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


def write_excel(case_table, excel_path):
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for split in ("train", "val", "test"):
            df_split = case_table.loc[case_table["split"] == split].copy()
            df_split.to_excel(writer, sheet_name=split, index=False)


def main():
    args = parse_args()

    model_dir = os.path.abspath(args.model_dir)
    meta = load_metadata(model_dir)

    case_table_path = (
        args.case_table
        if args.case_table
        else os.path.join(model_dir, meta.get("case_table_file", "case_table.csv"))
    )
    if not os.path.isfile(case_table_path):
        raise FileNotFoundError(f"Fant ikke case_table.csv: {case_table_path}")

    case_table = pd.read_csv(case_table_path, sep=";")
    tag = checkpoint_tag(args.checkpoint)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(model_dir, f"nnUNet_eval_{tag}")
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

    excel_path = os.path.join(output_dir, "data_splits.xlsx")
    summary_csv = os.path.join(output_dir, "dice_summary.csv")
    write_excel(scored, excel_path)
    summary.to_csv(summary_csv, sep=";", index=False)

    print(f"Excel med train/val/test Dice skrevet til: {excel_path}")
    print(f"Dice-oppsummering skrevet til: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
