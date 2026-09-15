#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standardisert treningsscript for endometriekreft-tumorsegmentering
- støtter 1-3 modaliteter (VIBE, T2, ADC)
"""

# ==========================================================
# Standard library
# ==========================================================
import os
import datetime
import warnings
import shutil
import argparse

# ==========================================================
# Numerical / data
# ==========================================================
import numpy as np
import pandas as pd
import nibabel as nib

# ==========================================================
# PyTorch / TorchIO
# ==========================================================
import torch
import torchio as tio
from torch.utils.data import DataLoader

# ==========================================================
# MONAI
# ==========================================================
from monai.networks.nets import UNet

# ==========================================================
# fastai
# ==========================================================
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import fit_one_cycle

# ==========================================================
# sklearn
# ==========================================================
from sklearn.model_selection import train_test_split

# ==========================================================
# Project-specific
# ==========================================================
import params
from datasetgenerator import (
    datasetgenerator_bergen,
    datasetgenerator_hong_kong,
    datasetgenerator_mont,
)
from utils import (
    build_image_path,
    build_mask_path,
    compute_tumor_volume,
    get_dataloader,
    monai_unet_model,
    nnunet_resenc_l_model,
    ssm_model,
    DiceMetric,
    DiceLossBinary,      # eller DiceBCELoss
    save_training_artifacts,
    save_training_plot,
    evaluate_model_and_compute_dice,
    debug_plot_first_batch,
)
from utils_finetuning import (
    build_finetuned_model_name,
    get_freeze_modes,
    prepare_model_for_finetuning,
)


def summarize_dice_scores(split_scores):
    rows = []
    for split, scores in split_scores.items():
        values = np.asarray(scores, dtype=float)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        rows.append(
            {
                "split": split,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "mean_std": f"{np.mean(values):.3f} ({np.std(values, ddof=1) if values.size > 1 else 0.0:.3f})",
                "median_iqr": f"{median:.3f} ({q1:.3f}-{q3:.3f})",
            }
        )
    return pd.DataFrame(rows)

# ==========================================================
# Argument parser
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MONAI 3D UNet for EC tumor segmentation"
    )

    # --- Modaliteter ---
    parser.add_argument(
        "--modalities",
        nargs="+",
        required=True,
        help="Liste over modaliteter, f.eks: T2 ADC vibe2min",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Referansemodalitet (f.eks T2 eller vibe2min)",
    )

    # --- Data split ---
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)

    # --- Trening ---
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--model_name",
        type=str,
        default="UNet",
        choices=["UNet", "nnUNet", "SSM"],
        help=(
            "Modellarkitektur: MONAI UNet, nnU-Net v2-inspirert ResEnc L, "
            "eller SSM/Mamba-inspirert 3D-modell."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bergen"],
        choices=["bergen", "mont", "hong_kong"],
        help="Hvilke datasett som skal brukes i trening (default: bergen)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Sti til eksisterende modellmappe for finetuning. Forventer best_model.pth inni mappen.",
    )
    parser.add_argument(
        "--freeze_mode",
        type=str,
        default="none",
        choices=get_freeze_modes(),
        help=(
            "Hvilke deler av pretrained modell som skal trenes videre under finetuning. "
            "For nnUNet: none=trener hele modellen; "
            "head=trener bare decoder.seg_layers; "
            "decoder=trener hele decoder; "
            "decoder_bottleneck=trener decoder + encoder.stages.5. "
            "For SSM: head=bare segmenteringshodet; decoder=decoder+head; "
            "decoder_bottleneck=decoder+head+bottleneck. "
            "For UNet brukes tilsvarende MONAI-UNet-deler."
        ),
    )

    

    return parser.parse_args()

    
# ==========================================================
# Main
# ==========================================================
def main():
    args = parse_args()

    pd.set_option("display.max_rows", None)
    selected_modalities = args.modalities
    reference = args.reference

    # ------------------------------
    # GPU
    # ------------------------------
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Bruker GPU {device}")
    torch.backends.cudnn.benchmark = True

    # ------------------------------
    # Datasett
    # ------------------------------    
    datasets = args.datasets
    print(f"Bruker datasett: {datasets}")
    
    warnings.filterwarnings(
        "ignore",
        message="Using TorchIO images without a torchio.SubjectsLoader",
    )

    # ==========================================================
    # 1. Les og filtrer datasett
    # ==========================================================
    selected_modalities = args.modalities
    datasets = args.datasets
    
    # --------------------------------------------------
    # Valider datasett + modaliteter
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Valider referanse
    # --------------------------------------------------
    if reference not in selected_modalities:
        raise ValueError(
            f"Reference '{reference}' må være en av valgte modaliteter: "
            f"{selected_modalities}"
        )    
    print(f"Datasett: {datasets}")
    print(f"Modaliteter: {selected_modalities}")
    print(f"Referanse: {reference}")

    #    df = pd.read_csv(params.pathlistvalid, sep=";")

    # Local dataset
#    require = [params.modalities[m]["col"] for m in selected_modalities]
#    df = datasetgenerator(df, require)
    
    dfs = []
    # --------------------------------------------------
    # Bergen-datasett
    # --------------------------------------------------
    if "bergen" in datasets:
        print(f" Leser Bergen-datasett {params.pathlistvalid_bergen}")
        df_bergen = pd.read_csv(params.pathlistvalid_bergen, sep=";").groupby("subj", as_index=False).first()
        mod_info = params.DATASET_CONFIG["bergen"]["modalities"]
        require = [mod_info[m]["col"] for m in selected_modalities]
        print(f"Krever verdier for {require}")
        df_bergen = datasetgenerator_bergen(df_bergen, require)
        df_bergen["source"] = "bergen"    
        dfs.append(df_bergen)

    # --------------------------------------------------
    # Mont-datasett
    # --------------------------------------------------
    if "mont" in datasets:
        print(f" Leser Mont-datasett {params.pathlistvalid_mont}")    
        df_mont = pd.read_csv(params.pathlistvalid_mont, sep=";").groupby("subj", as_index=False).first() # <- egen CSV    
        mod_info = params.DATASET_CONFIG["mont"]["modalities"]
        require = [mod_info[m]["col"] for m in selected_modalities]    
        print(f"Krever verdier for {require}")
        df_mont = datasetgenerator_mont(df_mont, require)
        df_mont["source"] = "mont"
        dfs.append(df_mont)

    # --------------------------------------------------
    # Hong Kong-datasett
    # --------------------------------------------------
    if "hong_kong" in datasets:
        print(f" Leser Hong-Kong-datasett {params.pathlistvalid_hong_kong}")
        df_hong_kong = pd.read_csv(params.pathlistvalid_hong_kong, sep=";").groupby("subj", as_index=False).first()
        mod_info = params.DATASET_CONFIG["hong_kong"]["modalities"]
        require = [mod_info[m]["col"] for m in selected_modalities]
        print(f"Krever verdier for {require}")
        df_hong_kong = datasetgenerator_hong_kong(df_hong_kong, require)
        df_hong_kong["source"] = "hong_kong"
        dfs.append(df_hong_kong)

    # --------------------------------------------------
    # Slå sammen
    # --------------------------------------------------
    if len(dfs) == 0:
        raise RuntimeError("Ingen datasett valgt - dette skal ikke skje")
    
    df = pd.concat(dfs, ignore_index=True)
            
    # Bare manuelle datasett
    df = df.loc[df.dataset == "man"].reset_index(drop=True)
    print(f"Antall datasett med manuelle masker: {len(df)}")

    print("Antall rader per datasett:")
    print(df.source.value_counts())
    
    # ==========================================================
    # 2. Bygg paths (PER RAD, basert på source)
    # ==========================================================
    
    df["imgpath"] = [
        build_image_path(
            subj=s,
            modalities=selected_modalities,
            prepathnii=getattr(params, f"prepathnifti_{src}"),
            reference=reference,
            dataset=src,
        )
        for s, src in zip(df.subj, df.source)
    ]

    df["pathmask"] = [
        build_mask_path(
            subj=s,
            maskname=m,
            prepathnii=getattr(params, f"prepathnifti_{src}"),
            dataset=src, 
        )
        for s, m, src in zip(df.subj, df.pathmask, df.source)
    ]
    
    print("\nDEBUG: Dimensjoner på alle filer i imgpath\n")
    for idx, row in df.iterrows():
        print(f"Subj: {row.subj} | Dataset: {row.source}")
        paths = row.imgpath.split(";")
    
        for p in paths:
            if not os.path.exists(p):
                print(f"  Mangler fil: {p}")
                continue
    
            img = nib.load(p)
            shape = img.shape
    
            print(f"  {os.path.basename(p)}, {shape}")
        print("-" * 60)

    # ==========================================================
    # 3. Tumorvolum + stratified split
    # ==========================================================
    print("Beregner tumorvolum ...")
    df["tumorsize"] = [
        compute_tumor_volume(p) for p in df["pathmask"].values
    ]

    df["tumorsize_cat"] = pd.qcut(
        df["tumorsize"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
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
        f"Train: {len(dftrain)}, "
        f"Val: {dftrain['isval'].sum()}, "
        f"Test: {len(dftest)}"
    )

    # ==========================================================
    # 4. Dataloaders
    # ==========================================================
    train_dl = get_dataloader(
        dftrain[dftrain.isval == False],
        args.batch_size,
        args.img_size,
        selected_modalities,
        reference,
        augmentation="basic",
    )

    val_dl = get_dataloader(
        dftrain[dftrain.isval == True],
        args.batch_size,
        args.img_size,
        selected_modalities,
        reference,
        augmentation="none",
    )

    # ==========================================================
    # DEBUG PLOT - slå av/på ved behov
    # ==========================================================
    DEBUG_PLOT = False
    if DEBUG_PLOT:
        debug_plot_first_batch(
            dataloader=train_dl,
            modality_names=selected_modalities,
            max_items=6,
        )    
        return  # stopper før trening
        
    dls = DataLoaders(train_dl, val_dl, device=device)
    dls.n_inp = 1

    # For lagring
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    modalities_str = "_".join(selected_modalities)
    datasets_str = "+".join(sorted(datasets))
    if args.pretrained:
        basename = build_finetuned_model_name(args.pretrained, datasets)
    else:
        basename = (
            f"{args.model_name}_{modalities_str}_{params.version}_"
            f"{datasets_str}_{timestamp}"
        )
    save_dir = os.path.join(params.prepathmodels, basename)
    os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================================
    # 5. Modell
    # ==========================================================
    if args.model_name == "UNet":
        model = monai_unet_model(
            in_channels=len(selected_modalities),
            dropout=args.dropout,
        ).to(device)
    elif args.model_name == "nnUNet":
        if args.dropout != 0.1:
            print("nnUNet bruker ikke dropout; --dropout ignoreres.")
        model = nnunet_resenc_l_model(
            in_channels=len(selected_modalities),
        ).to(device)
    elif args.model_name == "SSM":
        model = ssm_model(
            in_channels=len(selected_modalities),
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Ukjent model_name: {args.model_name}")

    if args.pretrained:
        prepare_model_for_finetuning(
            model=model,
            model_dir=args.pretrained,
            device=device,
            freeze_mode=args.freeze_mode,
            model_name=args.model_name,
        )

        
    metric = DiceMetric()

    learn = Learner(
        dls,
        model,
        loss_func=DiceLossBinary(),
        metrics=[metric],
        path=save_dir,     # VIKTIG
        model_dir=".",     # lagre rett i save_dir
        cbs=[
            MixedPrecision(), 
            SaveModelCallback(
                monitor="dice",
                comp=np.greater,
                fname="best_model",
            ),
            EarlyStoppingCallback(
                monitor="dice",
                comp=np.greater,
                patience=args.patience,
            ),
        ],
    )
    print("TYPE learn:", type(learn))
    print("TYPE model:", type(model))

    # ==========================================================
    # 6. Trening
    # ==========================================================
    learn.fit_one_cycle(
        args.epochs,
        lr_max=args.lr,
        wd=args.weight_decay,
    )    
    learn.load("best_model")
    
    # ==========================================================
    # 6b. Lagre Dice per epoch til CSV
    # ==========================================================
    rec = learn.recorder
    values = np.array(rec.values)  # shape: (n_epochs, 2 + n_metrics)
    epochs = np.arange(1, len(values) + 1)

    # Kolonnenavn: train_loss, valid_loss, + metrics
    metric_names = []
    for m in learn.metrics:
        metric_names.append(getattr(m, "name", m.__class__.__name__))
    cols = ["epoch", "train_loss", "valid_loss"] + metric_names
    df_metrics = pd.DataFrame(
        np.column_stack([epochs, values]),
        columns=cols
    )

    # Lagre til CSV
    metrics_csv = os.path.join(save_dir, "training_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False, sep=';')
    print(f"Dice og loss lagret til: {metrics_csv}")
        

    # ==========================================================
    # Prediker på alle data
    # ==========================================================
    
    print('Applying the model')
    # Anvend på train
    train_eval_dl = get_dataloader(
        dftrain[dftrain.isval == False],
        args.batch_size,
        args.img_size,
        selected_modalities,
        reference,
        augmentation="none",
    )
    train_dice = evaluate_model_and_compute_dice(
        model,
        train_eval_dl,
        device,
    )
    dftrain.loc[dftrain.isval == False, "dice"] = train_dice

    # Anvend på val
    val_eval_dl = get_dataloader(
        dftrain[dftrain.isval == True],
        args.batch_size,
        args.img_size,
        selected_modalities,
        reference,
        augmentation="none",
    )
    
    val_dice = evaluate_model_and_compute_dice(
        model,
        val_eval_dl,
        device,
    )    
    dftrain.loc[dftrain.isval == True, "dice"] = val_dice

    # Anvend på test
    test_eval_dl = get_dataloader(
        dftest,
        args.batch_size,
        args.img_size,
        selected_modalities,
        reference,
        augmentation="none",
    )
    
    test_dice = evaluate_model_and_compute_dice(
        model,
        test_eval_dl,
        device,
    )
    dftest["dice"] = test_dice

    # Check validity
    assert len(train_dice) == (dftrain.isval == False).sum()
    assert len(val_dice) == (dftrain.isval == True).sum()
    assert len(test_dice) == len(dftest)

    dice_summary = summarize_dice_scores(
        {
            "train": train_dice,
            "val": val_dice,
            "test": test_dice,
        }
    )
    dice_summary_csv = os.path.join(save_dir, "dice_summary.csv")
    dice_summary.to_csv(dice_summary_csv, index=False, sep=";")
    print(f"Dice-oppsummering lagret til: {dice_summary_csv}")

    # ==========================================================
    # 7. Lagring
    # ==========================================================

    #model_path = os.path.join(save_dir, "model_best.pth")
    #torch.save(model.state_dict(), model_path)
    model_path = os.path.join(save_dir, "best_model.pth")
    save_training_artifacts(
        save_dir=save_dir,
        basename=basename,
        timestamp=timestamp,
        batch_size=args.batch_size,
        img_size=args.img_size,
        modalities=selected_modalities,
        reference=reference,
        train_df=dftrain[dftrain.isval == False],
        val_df=dftrain[dftrain.isval == True],
        test_df=dftest,
        model_path=model_path,
        modelname=args.model_name,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        datasets=datasets,
        pretrained_model_dir=args.pretrained,
        freeze_mode=args.freeze_mode,
    )
    print(f"Ferdig. Resultater lagret i:\n{save_dir}")

    # Save plots of losses
    print("DEBUG df_metrics columns:", df_metrics.columns.tolist())
    save_training_plot(df_metrics, save_dir)


# ==========================================================
# Entrypoint
# ==========================================================
if __name__ == "__main__":
    main()
