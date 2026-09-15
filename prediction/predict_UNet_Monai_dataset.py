#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import torch

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
    build_image_path,
    build_mask_path,
    compute_dice_score,
    get_dataloader,
    monai_unet_model,
    nnunet_resenc_l_model,
    read_hf_token_file,
    resolve_model_dir_from_hf,
    swinunetr_model,
    ssm_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MONAI U-Net prediction for all cases in a dataset table.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Case selection:\n"
            "  Use exactly one of:\n"
            "    --datasets bergen [mont ...]\n"
            "    --case_table cases.xlsx\n\n"
            "case_table columns:\n"
            "  Required minimum:\n"
            "    subj\n\n"
            "  Required when imgpath/pathmask cannot be looked up from a local model table:\n"
            "    subj\n"
            "    imgpath\n"
            "    pathmask\n\n"
            "  Optional:\n"
            "    split      Used for split-wise summary and --split filtering.\n"
            "    case_id    Only relevant for true nnUNetv2; generated locally if missing.\n\n"
            "imgpath format for multi-channel models:\n"
            "  One path per channel, separated by semicolons, in the same order as\n"
            "  modalities in settings.json.\n\n"
            "  Example for modalities [vibe2min, T2, ADC]:\n"
            "    /path/vibe2min.nii.gz;/path/T2.nii.gz;/path/ADC.nii.gz\n"
        ),
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to model directory, Hugging Face repo id, or huggingface.co URL",
    )
    parser.add_argument(
        "--hf_revision",
        default=None,
        help="Optional Hugging Face revision/branch/tag/commit.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or huggingface-cli login.",
    )
    parser.add_argument(
        "--hf_token_file",
        default=None,
        help="Optional file containing a Hugging Face token. Ignored if --hf_token is set.",
    )
    parser.add_argument(
        "--case_table",
        default=None,
        help=(
            "CSV or Excel file with cases to predict/evaluate. Must contain at least "
            "a subj column. If imgpath is needed, use one path per channel separated "
            "by semicolons in model modality order, for example "
            "'/path/vibe2min.nii.gz;/path/T2.nii.gz;/path/ADC.nii.gz'."
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
        "--out_dir",
        default=None,
        help="Output directory. Defaults to prediction/ under the local model folder.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Optional split to select when the dataset table has a split column.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary mask",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index. Defaults to CPU unless provided and CUDA is available.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation. Defaults to batch_size from settings.json.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    return parser.parse_args()


def load_settings(model_dir):
    settings_path = os.path.join(model_dir, "settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Could not find settings.json: {settings_path}")

    with open(settings_path) as f:
        return json.load(f)


def run_nnunetv2_wrapper(args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "evaluate_nnUNet_v2.py"),
        "--model_dir",
        args.model_dir,
        "--output_prefix",
        os.path.splitext(os.path.basename(__file__))[0],
    ]
    if args.hf_revision:
        cmd.extend(["--hf_revision", args.hf_revision])
    if args.hf_cache_dir:
        cmd.extend(["--hf_cache_dir", args.hf_cache_dir])
    if args.hf_token:
        cmd.extend(["--hf_token", args.hf_token])
    if args.hf_token_file:
        cmd.extend(["--hf_token_file", args.hf_token_file])
    if args.case_table:
        cmd.extend(["--case_table", args.case_table])
    if args.datasets:
        cmd.extend(["--datasets", *args.datasets])
    if args.out_dir:
        cmd.extend(["--out_dir", args.out_dir])
    if args.split:
        cmd.extend(["--split", args.split])
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])

    subprocess.run(cmd, check=True)


def safe_folder_name(value):
    value = str(value).strip().replace("hf://", "")
    value = value.replace("/", "__").replace("\\", "__")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def resolve_output_model_dir(input_model_ref, resolved_model_dir, meta):
    if os.path.exists(os.path.expanduser(input_model_ref)):
        return resolved_model_dir

    model_name = meta.get("model_name") or os.path.basename(
        input_model_ref.rstrip("/")
    )
    local_model_dir = os.path.join(params.prepathmodels, safe_folder_name(model_name))
    os.makedirs(local_model_dir, exist_ok=True)

    src_settings = os.path.join(resolved_model_dir, "settings.json")
    dst_settings = os.path.join(local_model_dir, "settings.json")
    if os.path.isfile(src_settings) and not os.path.exists(dst_settings):
        shutil.copy2(src_settings, dst_settings)

    return local_model_dir


def load_model(model_dir, device, meta=None):
    if meta is None:
        meta = load_settings(model_dir)

    modalities = list(meta["modalities"])
    dropout = float(meta.get("dropout", 0.0))
    model_kwargs = meta.get("model_kwargs", {})
    model_file = meta.get("model_file", "best_model.pth")
    model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file: {model_path}")

    model_architecture = meta.get("model_architecture", meta.get("modelname", "UNet"))
    if model_architecture == "UNet":
        model = monai_unet_model(
            in_channels=len(modalities),
            dropout=dropout,
        )
    elif model_architecture == "SSM":
        model = ssm_model(
            in_channels=len(modalities),
            dropout=dropout,
        )
    elif model_architecture == "nnUNet":
        model = nnunet_resenc_l_model(
            in_channels=len(modalities),
        )
    elif model_architecture == "SwinUNETR":
        model = swinunetr_model(
            in_channels=len(modalities),
            img_size=int(meta["img_size"]),
            dropout=dropout,
            use_checkpoint=False,
            feature_size=int(model_kwargs.get("feature_size", 24)),
        )
    elif model_architecture == "nnUNetv2":
        raise ValueError(
            "Ekte nnU-Net v2 evalueres med prediction/evaluate_nnUNet_v2.py, "
            "ikke predict_UNet_Monai_dataset.py."
        )
    else:
        raise ValueError(f"Ukjent model_architecture='{model_architecture}'")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, meta


def read_dataset_table(path, split="all"):
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
        build_image_path(
            subj=subj,
            modalities=modalities,
            prepathnii=getattr(params, f"prepathnifti_{source}"),
            reference=reference,
            dataset=source,
        )
        for subj, source in zip(df.subj, df.source)
    ]
    df["pathmask"] = [
        build_mask_path(
            subj=subj,
            maskname=maskname,
            prepathnii=getattr(params, f"prepathnifti_{source}"),
            dataset=source,
        )
        for subj, maskname, source in zip(df.subj, df.pathmask, df.source)
    ]
    return df


def evaluate_dataloader(model, dataloader, device, threshold):
    model.eval()
    all_dice = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            dice = compute_dice_score(
                preds,
                yb,
                apply_threshold=True,
                threshold=threshold,
            )
            all_dice.extend(dice.cpu().tolist())
    return all_dice


def validate_imgpaths(df, modalities):
    expected = len(modalities)
    bad = []
    for _, row in df.iterrows():
        n_paths = len(str(row["imgpath"]).split(";"))
        if n_paths != expected:
            bad.append((row.get("subj", ""), n_paths))
    if bad:
        details = ", ".join(f"{subj}: {n}" for subj, n in bad[:10])
        raise ValueError(
            f"Expected {expected} modality paths per case, but found mismatches: "
            f"{details}"
        )


def print_summary(results):
    dice = results["dice"].dropna()
    print("\n=== Dice summary ===")
    print(f"n          : {len(dice)}")
    print(f"mean       : {dice.mean():.4f}")
    print(f"median     : {dice.median():.4f}")
    print(f"std        : {dice.std(ddof=1):.4f}")
    print(f"min        : {dice.min():.4f}")
    print(f"max        : {dice.max():.4f}")
    print(f"q25        : {dice.quantile(0.25):.4f}")
    print(f"q75        : {dice.quantile(0.75):.4f}")


def summarize_results(results, run_id, model_ref, model_dir, dataset_path, split, threshold):
    dice = results["dice"].dropna()
    summary = {
        "run_id": run_id,
        "model_ref": model_ref,
        "model_dir": model_dir,
        "dataset": dataset_path,
        "split": split,
        "threshold": float(threshold),
        "n": int(len(dice)),
        "mean": float(dice.mean()),
        "median": float(dice.median()),
        "std": float(dice.std(ddof=1)) if len(dice) > 1 else 0.0,
        "min": float(dice.min()),
        "max": float(dice.max()),
        "q25": float(dice.quantile(0.25)),
        "q75": float(dice.quantile(0.75)),
    }
    summary["iqr"] = summary["q75"] - summary["q25"]
    summary["mean_std"] = f"{summary['mean']:.4f} ({summary['std']:.4f})"
    summary["median_iqr"] = (
        f"{summary['median']:.4f} "
        f"({summary['q25']:.4f}-{summary['q75']:.4f})"
    )
    return pd.DataFrame([summary])


def dice_summary_stats(dice):
    dice = pd.Series(dice).dropna()
    stats = {
        "n": int(len(dice)),
        "mean": float(dice.mean()),
        "median": float(dice.median()),
        "std": float(dice.std(ddof=1)) if len(dice) > 1 else 0.0,
        "min": float(dice.min()),
        "max": float(dice.max()),
        "q25": float(dice.quantile(0.25)),
        "q75": float(dice.quantile(0.75)),
    }
    stats["mean_std"] = f"{stats['mean']:.4f} ({stats['std']:.4f})"
    stats["median_iqr"] = (
        f"{stats['median']:.4f} "
        f"({stats['q25']:.4f}-{stats['q75']:.4f})"
    )
    return stats


def write_split_summaries(f, results):
    if "split" not in results.columns:
        return

    f.write("\nDice by split\n")
    f.write("=============\n")
    for split_value, group in results.groupby("split", dropna=False):
        stats = dice_summary_stats(group["dice"])
        f.write(f"\nSplit       : {split_value}\n")
        f.write(f"n           : {stats['n']}\n")
        f.write(f"Mean (std)  : {stats['mean_std']}\n")
        f.write(f"Median (IQR): {stats['median_iqr']}\n")
        f.write(f"Min-max     : {stats['min']:.4f}-{stats['max']:.4f}\n")


def print_per_case_dice(results):
    print("\n=== Dice per case ===")
    for _, row in results.iterrows():
        print(f"subj {row['subj']}: dice={row['dice']:.4f}")


def default_dataset_path(model_dir):
    dataset_path = os.path.join(model_dir, "data_splits.xlsx")
    if os.path.exists(dataset_path):
        return dataset_path
    legacy_dftest = os.path.join(model_dir, "dftest.csv")
    if os.path.exists(legacy_dftest):
        return legacy_dftest
    return dataset_path


def complete_case_table(df, dataset_path, model_dir, split):
    if "subj" not in df.columns:
        raise ValueError(f"Missing required column in {dataset_path}: ['subj']")

    required_for_eval = {"imgpath", "pathmask"}
    missing_for_eval = required_for_eval.difference(df.columns)
    if not missing_for_eval:
        return df

    lookup_path = default_dataset_path(model_dir)
    if not os.path.exists(lookup_path):
        raise ValueError(
            f"Missing columns needed for evaluation in {dataset_path}: "
            f"{sorted(missing_for_eval)}. Could not find model data_splits.xlsx "
            f"or dftest.csv to look them up by subj."
        )

    lookup = read_dataset_table(lookup_path, split=split)
    if "subj" not in lookup.columns:
        raise ValueError(f"Lookup table is missing subj column: {lookup_path}")
    missing_in_lookup = required_for_eval.difference(lookup.columns)
    if missing_in_lookup:
        raise ValueError(
            f"Lookup table {lookup_path} is missing required columns: "
            f"{sorted(missing_in_lookup)}"
        )

    lookup_cols = ["subj"] + [
        c for c in lookup.columns if c not in df.columns and c in required_for_eval
    ]
    lookup = lookup[lookup_cols].drop_duplicates("subj")
    merged = df.merge(lookup, on="subj", how="left")

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


def main():
    args = parse_args()
    run_id = datetime.datetime.now().strftime("%Y%m%d")

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
    meta = load_settings(model_dir)
    if meta.get("model_architecture") == "nnUNetv2":
        run_nnunetv2_wrapper(args)
        return

    output_model_dir = resolve_output_model_dir(input_model_ref, model_dir, meta)

    modalities = list(meta["modalities"])
    reference = meta.get("reference", modalities[0])

    if args.datasets:
        dataset_path = "datasets=" + ",".join(args.datasets)
    else:
        dataset_path = args.case_table or default_dataset_path(model_dir)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Could not find dataset table: {dataset_path}. "
                "Use --case_table /path/to/cases.xlsx or data_splits.xlsx."
            )

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = os.path.join(output_model_dir, "prediction")

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    print(f"Model directory : {model_dir}")
    print(f"Output model dir: {output_model_dir}")
    print(f"Case source     : {dataset_path}")
    print(f"Split           : {args.split}")
    print(f"Results dir     : {out_dir}")
    print(f"Device          : {device}")

    if args.datasets:
        df = build_project_case_table(args.datasets, modalities, reference)
    else:
        df = read_dataset_table(dataset_path, split=args.split)
        if args.split != "all" and "split" in df.columns:
            df = df[df["split"].astype(str) == args.split].copy()
            if df.empty:
                raise ValueError(f"No rows with split='{args.split}' in {dataset_path}")

        df = complete_case_table(
            df=df,
            dataset_path=dataset_path,
            model_dir=model_dir,
            split=args.split,
        )

    df = df.sort_values("subj").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)

    model, meta = load_model(model_dir, device, meta=meta)

    img_size = int(meta["img_size"])
    batch_size = args.batch_size or int(meta.get("batch_size", 1))

    validate_imgpaths(df, modalities)

    print(f"Modalities      : {modalities}")
    print(f"Reference       : {reference}")
    print(f"Image size      : {img_size}")
    print(f"Batch size      : {batch_size}")
    print(f"Number of cases : {len(df)}")

    dataloader = get_dataloader(
        df=df,
        batch_size=batch_size,
        img_size=img_size,
        selected_modalities=modalities,
        reference=reference,
        augmentation="none",
        num_workers=args.num_workers,
    )

    dice = evaluate_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
    )
    if len(dice) != len(df):
        raise RuntimeError(f"Expected {len(df)} Dice values, got {len(dice)}")

    results = df.copy()
    results["dice"] = dice

    print_per_case_dice(results)

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    results_path = os.path.join(out_dir, f"{script_name}_results_{run_id}.xlsx")
    summary_path = os.path.join(out_dir, f"{script_name}_summary_{run_id}.txt")

    sheet_name = output_sheet_name(dataset_path, args.split)
    with pd.ExcelWriter(results_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\nSaved per-case results to: {results_path}")

    summary = summarize_results(
        results=results,
        run_id=run_id,
        model_ref=args.model_dir,
        model_dir=output_model_dir,
        dataset_path=dataset_path,
        split=args.split,
        threshold=args.threshold,
    )
    summary_row = summary.iloc[0]
    with open(summary_path, "w") as f:
        f.write("Dice summary\n")
        f.write("============\n")
        f.write(f"Run ID      : {summary_row['run_id']}\n")
        f.write(f"Model ref   : {summary_row['model_ref']}\n")
        f.write(f"Model dir   : {summary_row['model_dir']}\n")
        f.write(f"Dataset     : {summary_row['dataset']}\n")
        f.write(f"Split       : {summary_row['split']}\n")
        f.write(f"Threshold   : {summary_row['threshold']}\n")
        f.write(f"n           : {summary_row['n']}\n")
        f.write(f"Mean (std)  : {summary_row['mean_std']}\n")
        f.write(f"Median (IQR): {summary_row['median_iqr']}\n")
        f.write(f"Min-max     : {summary_row['min']:.4f}-{summary_row['max']:.4f}\n")
        write_split_summaries(f, results)
    print(f"Saved Dice summary to: {summary_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
