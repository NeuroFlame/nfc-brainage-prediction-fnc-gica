"""
Pure computation logic for the brainage FNC local SVR steps.
No NVFlare dependencies — all inputs and outputs are plain Python/numpy objects.
"""

import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR

from _utils.svr_utils import form_XYMatrices, get_metrics
from _utils.preprocessor_utils import split_xy_data


def load_site_data(data_dir: str, data_file: str, label_file: str,
                   input_source: str, split_type: str,
                   test_size: float, shuffle: bool) -> dict:
    """
    Load FNC data and covariates, form X/y matrices, and split into train/test.

    :param data_dir: Directory containing data_file and label_file.
    :param data_file: Filename of the .mat FNC data file.
    :param label_file: Filename of the covariates CSV (must contain an 'age' column).
    :param input_source: 'GICA' or 'UKBioBank_Comp2019'.
    :param split_type: 'random' or 'age_range_stratified'.
    :param test_size: Fraction of subjects reserved for testing.
    :param shuffle: Whether to shuffle before splitting.
    :return: Dict with keys X_train, X_test, y_train, y_test.
    """
    X, y = form_XYMatrices(data_dir, input_source, data_file, label_file)
    X_train, X_test, y_train, y_test = split_xy_data(split_type, X, y, test_size, shuffle)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def train_local_svr(X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray,
                    svr_params: dict) -> dict:
    """
    Train a MinMaxScaler + LinearSVR pipeline on all available local data
    (train + test merged), then evaluate on the original train/test splits.

    Non-owner sites call this in round 0. All local data is used for training
    so the maximum signal is contributed to the federated weight aggregation.

    :param X_train: Training feature matrix.
    :param X_test: Test feature matrix.
    :param y_train: Training labels (age).
    :param y_test: Test labels (age).
    :param svr_params: Keyword arguments for sklearn LinearSVR.
    :return: Dict with w_local, intercept_local, n_train/test, rmse/mae train/test.
    """
    X_all = np.vstack((X_train, X_test))
    y_all = np.hstack((y_train, y_test))

    pipeline = make_pipeline(
        preprocessing.MinMaxScaler(),
        LinearSVR(**svr_params),
    )
    pipeline.fit(X_all, y_all)

    svr_model = pipeline.named_steps["linearsvr"]
    w = np.squeeze(svr_model.coef_)
    intercept = svr_model.intercept_

    train_metrics = get_metrics(y_train, pipeline.predict(X_train))
    test_metrics = get_metrics(y_test, pipeline.predict(X_test))

    return {
        "w_local": w.tolist(),
        "intercept_local": intercept.tolist(),
        "n_train_samples_local": len(y_train),
        "n_test_samples_local": len(y_test),
        "rmse_train_local": float(train_metrics["rmse"]),
        "rmse_test_local": float(test_metrics["rmse"]),
        "mae_train_local": float(train_metrics["mae"]),
        "mae_test_local": float(test_metrics["mae"]),
    }


def train_owner_svr(X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray,
                    w_locals: list, svr_params: dict) -> dict:
    """
    Project owner data through the averaged local weights, then train a second
    LinearSVR on the projected (1-D) features.

    This is the owner site's round-1 computation. The projection U = X @ w_avg
    compresses the FNC feature space into a scalar that captures the shared
    cross-site signal, which the owner then re-fits.

    :param X_train: Owner training feature matrix (cached from round 0).
    :param X_test: Owner test feature matrix.
    :param y_train: Owner training labels.
    :param y_test: Owner test labels.
    :param w_locals: List of weight vectors from all non-owner sites, as received
                     from the server. Shape after np.array: (n_features, n_sites).
    :param svr_params: Keyword arguments for sklearn LinearSVR (owner variant).
    :return: Dict with w_owner, intercept_owner, n_train/test, rmse/mae train/test.
    """
    w_locals_arr = np.array(w_locals)          # (n_features, n_sites)
    w_avg = np.mean(w_locals_arr, axis=1).reshape(-1, 1)

    U_train = np.matmul(X_train, w_avg).astype(np.double)
    U_test = np.matmul(X_test, w_avg).astype(np.double)

    pipeline = make_pipeline(
        preprocessing.MinMaxScaler(),
        LinearSVR(**svr_params),
    )
    pipeline.fit(U_train, y_train)

    svr_model = pipeline.named_steps["linearsvr"]
    w_owner = np.squeeze(svr_model.coef_)
    fit_intercept = svr_params.get("fit_intercept", True)
    intercept_owner = float(svr_model.intercept_) if fit_intercept else 0.0

    train_metrics = get_metrics(y_train, pipeline.predict(U_train))
    test_metrics = get_metrics(y_test, pipeline.predict(U_test))

    return {
        "w_owner": w_owner.tolist(),
        "intercept_owner": intercept_owner,
        "n_train_samples_owner": len(U_train),
        "n_test_samples_owner": len(U_test),
        "rmse_train_owner": float(train_metrics["rmse"]),
        "rmse_test_owner": float(test_metrics["rmse"]),
        "mae_train_owner": float(train_metrics["mae"]),
        "mae_test_owner": float(test_metrics["mae"]),
    }
