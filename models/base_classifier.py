import numpy as np


class BaseClassifier:
    """Base class for all classifiers."""

    def __init__(self) -> None:
        """
        Initialize the classifier.
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier to the training data.

        Args:
            X (np.ndarray): The training features.
            y (np.ndarray): The training labels.

        Returns:
            None
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new data.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        pass

    def grid_search(
        self, X_train: np.ndarray, y_train: np.ndarray, param_grid: dict, cv: int = 5
    ) -> None:
        """
        Perform grid search cross-validation to find the best hyperparameters for the model.

        Args:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training labels.
            param_grid (dict): The parameter grid to search over.
            cv (int, optional): The number of folds for cross-validation. Defaults to 5.

        Returns:
            None: Updates the model with the best estimator found during the search.
        """
        pass

    def get_params(self) -> dict:
        """
        Get parameters of the model.

        Returns:
            dict: Parameter names mapped to their values.
        """
        pass
