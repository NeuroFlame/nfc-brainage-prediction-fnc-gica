"""Load site-local FNC matrices and ages."""

import os

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .types import SiteInputs

UKBIOBANK_SUBJECT_ID = "eid"
UKBIOBANK_AGE = "age_when_attended_assessment_centre_f21003_2_0"


def load_inputs(
    data_dir,
    input_source="GICA",
    data_file="coinstac-gica_postprocess_results.mat",
    label_file="covariates.csv",
) -> SiteInputs:
    """Build the feature matrix and age vector for this site."""
    if input_source == "GICA":
        X, y = _load_gica(data_dir, data_file, label_file)
    elif input_source == "UKBioBank_Comp2019":
        X, y = _load_ukbiobank(data_dir, data_file, label_file)
    else:
        raise ValueError(
            f"Unknown input_source {input_source!r}; use 'GICA' or 'UKBioBank_Comp2019'"
        )
    return SiteInputs(X=X, y=np.asarray(y, dtype=np.double).reshape(-1))


def upper_triangle_features(fnc_data: np.ndarray) -> np.ndarray:
    """Flatten each symmetric FNC matrix to its values above the diagonal."""
    rows, cols = np.triu_indices(fnc_data.shape[1], k=1)
    return fnc_data[:, rows, cols].astype(np.double)


def _load_gica(data_dir, data_file, label_file):
    # fnc_corrs_all is (subjects, sessions, components, components); use session 1.
    fnc_data = loadmat(os.path.join(data_dir, data_file))["fnc_corrs_all"][:, 0]
    covariate_df = pd.read_csv(os.path.join(data_dir, label_file))

    if len(fnc_data) != len(covariate_df):
        raise ValueError(
            f"Number of subjects in fnc_data ({len(fnc_data)}) and covariate "
            f"data ({len(covariate_df)}) do not match."
        )

    return upper_triangle_features(fnc_data), covariate_df["age"].to_numpy()


def _load_ukbiobank(data_dir, data_file, label_file):
    with h5py.File(os.path.join(data_dir, data_file), "r") as f:
        file_names = _cell_array_strings(f, "fN")[0]
        icn_ins = _matlab_array(f, "icn_ins")
        fnc_data = _matlab_array(f, "corrdata")

    eids = [int(name.split("/")[8].split("_")[0]) for name in file_names]
    df_eid = pd.DataFrame(eids, columns=[UKBIOBANK_SUBJECT_ID])
    df = pd.read_table(os.path.join(data_dir, label_file))
    df_eid_age = df[[UKBIOBANK_SUBJECT_ID, UKBIOBANK_AGE]]
    df_result = pd.concat([df_eid_age, df_eid], axis=1, join="inner").reindex(
        df_eid.index
    )

    # Keep only the intrinsic connectivity networks; MATLAB indices start at 1.
    icn = icn_ins.astype(int).reshape(len(icn_ins)) - 1
    fnc_data = fnc_data[:, icn][:, :, icn]

    return upper_triangle_features(fnc_data), df_result[UKBIOBANK_AGE].to_numpy()


def _cell_array_strings(hdf5_file, key_name):
    return [
        ["".join(map(chr, hdf5_file[ref][:])) for ref in column]
        for column in hdf5_file[key_name]
    ]


def _matlab_array(hdf5_file, key_name):
    # MATLAB v7.3 files store arrays column-major; reverse the axes.
    return np.array(hdf5_file[key_name]).T
