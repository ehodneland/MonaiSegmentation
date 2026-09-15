import os

# ==============================
# PATHS
# ==============================
prepath = '/raid/erlend/Dropbox/Precision_Imaging_in_Gynecologic_Cancer/EC/MonaiSegmentation'
prepathnifti_bergen = '/raid/erlend/GynKreft/Data-EC/Nifti'
prepathnifti_mont = '/raid/erlend/GynKreft/Data-EC/Montpelier/Nifti'
prepathnifti_hong_kong = '/path/to/Hong-Kong/Nifti'
pathlistvalid_bergen = '/raid/erlend/Dropbox/Precision_Imaging_in_Gynecologic_Cancer/EC/listAll/listvalid.csv'
pathlistvalid_mont = '/raid/erlend/Dropbox/Precision_Imaging_in_Gynecologic_Cancer/EC/Montpelier/data/listvalid.csv'
pathlistvalid_hong_kong = '/path/to/Hong-Kong/listvalid.csv'
prepathfigs = os.path.join(prepath, 'figs')
prepathdata = os.path.join(prepath, 'data')
prepathmodels = os.path.join(prepath, 'models')

# ==============================
# MODALITIES / INPUT CHANNELS
# ==============================
# params.py (eller utils/config)
MODALITY_KEYS = ["vibe2min", "T2", "ADC"]
DATASET_CONFIG = {

    "bergen": {
        "subject_prefix": "EC",
        "subject_format": "{prefix}{id:03d}",

        "modalities": {
            "vibe2min": {
                "col": "pathvibe2minDicom",
                "file": "vibe2min.nii.gz",
            },
            "T2": {
                "col": "pathT2Dicom",
                "file": "T2.nii.gz",
            },
            "ADC": {
                "col": "pathADCDicom",
                "file": "ADC.nii.gz",
            },
        },
    },

    "mont": {
        "subject_prefix": "",
        "subject_format": "{id}",   # eller "MONT_{id}"

        "modalities": {
            "vibe2min": {
                "col": "pathT1KDicom",
                "file": "T1K_2min.nii.gz",
            },
            "T2": {
                "col": "pathT2Dicom",
                "file": "T2.nii.gz",
            },
            "ADC": {
                "col": "pathADCDicom",
                "file": "ADC.nii.gz",
            },
        },
    },

    "hong_kong": {
        "subject_prefix": "",
        "subject_format": "{id}",   # TODO: oppdater ved faktisk subject-format

        "modalities": {
            "vibe2min": {
                "col": "pathT1KDicom",   # TODO: oppdater kolonnenavn
                "file": "T1K_2min.nii.gz",   # TODO: oppdater filnavn
            },
            "T2": {
                "col": "pathT2Dicom",    # TODO: oppdater kolonnenavn
                "file": "T2.nii.gz",     # TODO: oppdater filnavn hvis nødvendig
            },
            "ADC": {
                "col": "pathADCDicom",   # TODO: oppdater kolonnenavn
                "file": "ADC.nii.gz",    # TODO: oppdater filnavn hvis nødvendig
            },
        },
    },
}


# Modaliteter og tilhørende filnavn
#modalities = {
#    "vibe2min": {"col": "pathvibe2minDicom", "file": "vibe2min.nii.gz"},
#    "T2":   {"col": "pathT2Dicom",       "file": "T2.nii.gz"},
#    "ADC":  {"col": "pathADCDicom",      "file": "ADC.nii.gz"}
#}

version = 'v01'
