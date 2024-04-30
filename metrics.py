from typing import List
import numpy as np
import sklearn.metrics as metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return metrics.accuracy_score(y_true, y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return metrics.precision_score(y_true, y_pred)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return metrics.recall_score(y_true, y_pred)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return metrics.f1_score(y_true, y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    return metrics.confusion_matrix(y_true, y_pred)


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return metrics.roc_auc_score(y_true, y_pred)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return metrics.classification_report(y_true, y_pred)
