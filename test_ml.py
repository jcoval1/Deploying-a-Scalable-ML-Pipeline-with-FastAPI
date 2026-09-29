import os
import pickle
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from ml.model import compute_model_metrics, train_model, save_model


# TODO: implement the first test. Change the function name and input as needed
def test_compute_model_metrics_values():
    """
    Check that compute_model_metrics returns expected values
    """
    y = [1, 0, 1, 0, 0, 1, 0]
    preds = [1, 0, 1, 1, 0, 1, 0]
    p, r, fb = compute_model_metrics(y, preds)
    assert(p == 0.75)
    assert(r == 1)
    assert(0.857 < fb < 0.8572)


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_return_values():
    """
    Check that train)model returns a fitted LogisticRegression model
    """
    X_train = np.array([[0.5, 2.5, -0.2, 1.3, 0], [1, 2.5, 0.1, 1.0, -0.2], [0.7, 2.5, -0.1, 1.1, 1]])
    y_train = np.array([1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)
    check_is_fitted(model)


# TODO: implement the third test. Change the function name and input as needed
def test_save_model_writes(tmp_path):
    """
    Test whether the save_model function writes a file correctly
    """
    file_directory = tmp_path / "data_to_save.pkl"
    data_to_save = "We've gotten ourselves into quite a pickle"
    
    save_model(data_to_save, file_directory)
    assert file_directory.exists()
