#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ekte nnU-Net v2-wrapper for EC tumorsegmentering.

Scriptet bruker prosjektets eksisterende params.py og datasetgenerator.py for å
finne cases, modaliteter og masker, skriver et nnU-Net raw dataset, og kan
deretter kjøre nnU-Net v2 CLI:

  nnUNetv2_plan_and_preprocess
  nnUNetv2_train

Merk: selve nnU-Net v2 CLI krever vanligvis Python >=3.10. Bruk
--prepare_only i monai-miljøet hvis du bare vil bygge datasettet.
"""

import argparse
import datetime
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    compute_tumor_volume,
    create_raw_dataset_nnUNet,
    derive_dataset_id_nnUNet,
    format_dataset_name_nnUNet,
    get_nnunet_env_nnUNet,
    read_hf_token_file,
    require_nnunet_cli_nnUNet,
    resolve_model_dir_from_hf,
    resolve_pretrained_checkpoint_nnUNet,
    run_command_nnUNet,
    write_settings_nnUNet,
    write_splits_final_nnUNet,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and train a true nnU-Net v2 model using project data definitions."
    )

    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bergen"],
        choices=["bergen", "mont", "hong_kong"],
    )

    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument(
        "--dataset_id",
        type=int,
        default=None,
        help="Optional nnU-Net dataset ID. If omitted, a stable ID is derived automatically.",
    )
    parser.add_argument("--dataset_name", default="EC")
    parser.add_argument(
        "--nnunet_base_dir",
        default="",
        help=(
            "Optional override for the directory containing nnUNet_raw, "
            "nnUNet_preprocessed, and nnUNet_results. Default is "
            "<model_dir>/nnUNet_work so all files stay inside one model folder."
        ),
    )
    parser.add_argument(
        "--file_mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks or physical copies if --native_geometry is used.",
    )
    parser.add_argument(
        "--native_geometry",
        action="store_true",
        help=(
            "Do not resample modalities/masks to the reference modality before "
            "creating the nnU-Net raw dataset. Only use this if all channels "
            "already have identical geometry."
        ),
    )

    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--planner", default="nnUNetPlannerResEncL")
    parser.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--fold", default="0")
    parser.add_argument(
        "--pretrained",
        default="",
        help=(
            "Optional model directory, checkpoint file, or Hugging Face repo/URL for nnU-Net finetuning. "
            "If a directory is given, the script looks for --pretrained_checkpoint."
        ),
    )
    parser.add_argument("--hf_revision", default=None)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_token_file", default=None)
    parser.add_argument(
        "--pretrained_checkpoint",
        default="checkpoint_best.pth",
        help="Checkpoint name to load when --pretrained points to a directory.",
    )
    parser.add_argument(
        "--allow_same_results_dir",
        action="store_true",
        help=(
            "Allow finetuning to write into the same nnU-Net results folder as "
            "the pretrained checkpoint. Normally unsafe; prefer a new --dataset_id."
        ),
    )

    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only create nnU-Net raw dataset and metadata; do not run nnU-Net CLI.",
    )
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument(
        "--gpu",
        default="",
        help="Optional GPU id. Sets CUDA_VISIBLE_DEVICES for training and evaluation.",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip post-training evaluation with evaluate_nnUNet_v2.py.",
    )
    parser.add_argument(
        "--eval_checkpoint",
        default="checkpoint_best.pth",
        help="Checkpoint used for post-training evaluation.",
    )
    parser.add_argument("--eval_device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--eval_disable_tta", action="store_true")
    parser.add_argument("--verify_dataset_integrity", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    selected_modalities = args.modalities
    reference = args.reference
    datasets = args.datasets
    pretrained_dir = ""
    dataset_id = (
        args.dataset_id
        if args.dataset_id is not None
        else derive_dataset_id_nnUNet(args.dataset_name, selected_modalities, datasets)
    )
    pretrained_checkpoint = ""
    if args.pretrained:
        hf_token = args.hf_token or read_hf_token_file(args.hf_token_file)
        pretrained_dir = resolve_model_dir_from_hf(
            args.pretrained,
            revision=args.hf_revision,
            cache_dir=args.hf_cache_dir,
            token=hf_token,
        )
        pretrained_checkpoint = resolve_pretrained_checkpoint_nnUNet(
            pretrained_dir,
            checkpoint_name=args.pretrained_checkpoint,
        )
        print(f"Pretrained referanse: {args.pretrained}")
        print(f"Lokal pretrained mappe: {pretrained_dir}")
        print(f"Finetuning fra nnU-Net checkpoint: {pretrained_checkpoint}")

    warnings.filterwarnings(
        "ignore",
        message="Using TorchIO images without a torchio.SubjectsLoader",
    )

    for ds in datasets:
        if ds not in params.DATASET_CONFIG:
            raise ValueError(f"Ukjent datasett: {ds}")
        available = params.DATASET_CONFIG[ds]["modalities"].keys()
        for m in selected_modalities:
            if m not in available:
                raise ValueError(
                    f"Modalitet '{m}' finnes ikke i datasett '{ds}'. "
                    f"Tilgjengelig: {list(available)}"
                )

    if reference not in selected_modalities:
        raise ValueError(
            f"Reference '{reference}' må være en av valgte modaliteter: "
            f"{selected_modalities}"
        )

    dfs = []
    if "bergen" in datasets:
        print(f"Leser Bergen-datasett {params.pathlistvalid_bergen}")
        df_bergen = pd.read_csv(params.pathlistvalid_bergen, sep=";").groupby(
            "subj", as_index=False
        ).first()
        mod_info = params.DATASET_CONFIG["bergen"]["modalities"]
        require = [mod_info[m]["col"] for m in selected_modalities]
        df_bergen = datasetgenerator_bergen(df_bergen, require)
        df_bergen["source"] = "bergen"
        dfs.append(df_bergen)

    if "mont" in datasets:
        print(f"Leser Mont-datasett {params.pathlistvalid_mont}")
        df_mont = pd.read_csv(params.pathlistvalid_mont, sep=";").groupby(
            "subj", as_index=False
        ).first()
        mod_info = params.DATASET_CONFIG["mont"]["modalities"]
        require = [mod_info[m]["col"] for m in selected_modalities]
        df_mont = datasetgenerator_mont(df_mont, require)
        df_mont["source"] = "mont"
        dfs.append(df_mont)

    if "hong_kong" in datasets:
        print(f"Leser Hong-Kong-datasett {params.pathlistvalid_hong_kong}")
        df_hong_kong = pd.read_csv(params.pathlistvalid_hong_kong, sep=";").groupby(
            "subj", as_index=False
        ).first()
        mod_info = params.DATASET_CONFIG["hong_kong"]["modalities"]
        require = [mod_info[m]["col"] for m in selected_modalities]
        df_hong_kong = datasetgenerator_hong_kong(df_hong_kong, require)
        df_hong_kong["source"] = "hong_kong"
        dfs.append(df_hong_kong)

    if len(dfs) == 0:
        raise RuntimeError("Ingen datasett valgt")

    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[df.dataset == "man"].reset_index(drop=True)
    print(f"Antall datasett med manuelle masker: {len(df)}")
    print(df.source.value_counts())

    df["imgpath"] = [
        build_aligned_image_path_nnUNet(
            subj=s,
            modalities=selected_modalities,
            prepathnii=getattr(params, f"prepathnifti_{src}"),
            reference=reference,
            dataset=src,
        )
        for s, src in zip(df.subj, df.source)
    ]
    df["pathmask"] = [
        build_aligned_mask_path_nnUNet(
            subj=s,
            maskname=m,
            prepathnii=getattr(params, f"prepathnifti_{src}"),
            reference=reference,
            dataset=src,
        )
        for s, m, src in zip(df.subj, df.pathmask, df.source)
    ]

    print("Beregner tumorvolum ...")
    df["tumorsize"] = [compute_tumor_volume(p) for p in df["pathmask"].values]
    df["tumorsize_cat"] = pd.qcut(
        df["tumorsize"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    df = df.dropna(subset=["tumorsize_cat"]).reset_index(drop=True)

    dftrain, dftest = train_test_split(
        df,
        test_size=args.test_fraction,
        stratify=df["tumorsize_cat"],
        random_state=args.random_state,
    )

    val_idx = dftrain.sample(
        frac=args.validation_fraction,
        random_state=args.random_state,
    ).index
    dftrain["isval"] = False
    dftrain.loc[val_idx, "isval"] = True

    print(
        f"Train: {(dftrain.isval == False).sum()}, "
        f"Val: {dftrain.isval.sum()}, "
        f"Test: {len(dftest)}"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    modalities_str = "_".join(selected_modalities)
    datasets_str = "+".join(sorted(datasets))
    dataset_folder_name = format_dataset_name_nnUNet(dataset_id, args.dataset_name)
    basename = (
        f"nnUNetv2_{modalities_str}_{params.version}_"
        f"{datasets_str}_{dataset_folder_name}_{timestamp}"
    )
    if args.pretrained:
        source_name = os.path.basename(os.path.normpath(str(args.pretrained).rstrip("/")))
        basename = f"{basename}_finetuned_from_{source_name}"
    save_dir = os.path.join(params.prepathmodels, basename)
    os.makedirs(save_dir, exist_ok=True)
    nnunet_base_dir = (
        os.path.abspath(args.nnunet_base_dir)
        if args.nnunet_base_dir
        else os.path.join(save_dir, "nnUNet_work")
    )
    os.makedirs(nnunet_base_dir, exist_ok=True)

    raw_info = create_raw_dataset_nnUNet(
        dftrain=dftrain,
        dftest=dftest,
        modalities=selected_modalities,
        dataset_id=dataset_id,
        dataset_name=args.dataset_name,
        nnunet_base_dir=nnunet_base_dir,
        file_mode=args.file_mode,
        reference_idx=selected_modalities.index(reference),
        resample_to_reference=not args.native_geometry,
    )

    settings_path = write_settings_nnUNet(
        save_dir=save_dir,
        model_name=basename,
        dataset_id=dataset_id,
        dataset_folder_name=dataset_folder_name,
        modalities=selected_modalities,
        reference=reference,
        datasets=datasets,
        configuration=args.configuration,
        planner=args.planner,
        plans=args.plans,
        trainer=args.trainer,
        fold=args.fold,
        nnunet_base_dir=nnunet_base_dir,
        raw_dataset_dir=raw_info["dataset_dir"],
        case_table_path=raw_info["case_table_path"],
        pretrained_model_dir=args.pretrained,
        pretrained_checkpoint=pretrained_checkpoint,
    )

    print(f"nnU-Net raw dataset skrevet til: {raw_info['dataset_dir']}")
    print(f"Wrapper metadata skrevet til: {settings_path}")

    if args.prepare_only:
        print("prepare_only=True, stopper før nnU-Net CLI.")
        return

    require_nnunet_cli_nnUNet()
    env, nnunet_paths = get_nnunet_env_nnUNet(nnunet_base_dir)
    if args.gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.skip_preprocess:
        preprocess_cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(dataset_id),
            "-pl",
            args.planner,
            "-c",
            args.configuration,
            "--clean",
        ]
        if args.verify_dataset_integrity:
            preprocess_cmd.append("--verify_dataset_integrity")
        run_command_nnUNet(preprocess_cmd, env=env)

    preprocessed_dataset_dir = os.path.join(
        nnunet_paths["nnUNet_preprocessed"],
        dataset_folder_name,
    )
    splits_path = write_splits_final_nnUNet(
        preprocessed_dataset_dir,
        train_cases=raw_info["train_cases"],
        val_cases=raw_info["val_cases"],
    )
    print(f"nnU-Net split skrevet til: {splits_path}")

    if not args.skip_train:
        current_results_dir = os.path.join(
            nnunet_paths["nnUNet_results"],
            dataset_folder_name,
            f"{args.trainer}__{args.plans}__{args.configuration}",
            f"fold_{args.fold}",
        )
        if (
            pretrained_checkpoint
            and os.path.commonpath(
                [os.path.abspath(pretrained_checkpoint), os.path.abspath(current_results_dir)]
            )
            == os.path.abspath(current_results_dir)
            and not args.allow_same_results_dir
        ):
            raise RuntimeError(
                "Finetuning vil skrive til samme nnU-Net results-mappe som "
                "checkpointen du laster fra. Det kan overskrive originalmodellen. "
                "Bruk en ny --dataset_id for finetuning, eller sett "
                "--allow_same_results_dir hvis du virkelig ønsker dette."
            )

        train_cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            args.configuration,
            str(args.fold),
            "-tr",
            args.trainer,
            "-p",
            args.plans,
        ]
        if pretrained_checkpoint:
            train_cmd.extend(["-pretrained_weights", pretrained_checkpoint])
        run_command_nnUNet(train_cmd, env=env)

    if not args.skip_evaluation:
        if args.skip_train:
            print("skip_train=True, hopper over automatisk evaluering.")
        else:
            eval_script = os.path.join(
                REPO_ROOT,
                "prediction",
                "evaluate_nnUNet_v2.py",
            )
            eval_cmd = [
                sys.executable,
                eval_script,
                "--model_dir",
                save_dir,
                "--checkpoint",
                args.eval_checkpoint,
                "--device",
                args.eval_device,
            ]
            if args.gpu != "":
                eval_cmd.extend(["--gpu", str(args.gpu)])
            if args.eval_disable_tta:
                eval_cmd.append("--disable_tta")
            run_command_nnUNet(eval_cmd, env=env)

    print(f"Ferdig. Wrapper-resultater ligger i:\n{save_dir}")


if __name__ == "__main__":
    main()
