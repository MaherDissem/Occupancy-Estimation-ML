import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from models.base_classifier import BaseClassifier


class KNNClassifier(BaseClassifier):
    def __init__(self) -> None:
        self.model = KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
        )

        self.param_grid = {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "p": [1, 2],
            "metric": ["minkowski", "euclidean", "manhattan"],
        }

    def fit(self, X: np.ndarray, y: np.ndarray, run_grid_search=False) -> None:
        if run_grid_search:
            self.grid_search(X, y, self.param_grid)
        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def grid_search(
        self, X_train: np.ndarray, y_train: np.ndarray, param_grid: dict, cv: int = 5
    ) -> None:
        grid_search = GridSearchCV(
            estimator=self.model, param_grid=param_grid, cv=cv, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

    def get_params(self) -> dict:
        return self.model.get_params()
