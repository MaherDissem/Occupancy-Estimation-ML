import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from models.base_classifier import BaseClassifier


class SVMclassifier(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC(
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=0.001,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None,
        )

        self.param_grid = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [3, 4, 5],
            "gamma": ["scale", "auto"],
            "coef0": [0.0, 0.1, 0.5, 1.0],
            "shrinking": [True, False],
            "probability": [True, False],
            "tol": [0.001, 0.01, 0.1],
            "cache_size": [200, 400, 600],
            "class_weight": [None, "balanced"],
            "verbose": [False],
            "max_iter": [-1, 1000, 2000],
            "decision_function_shape": ["ovo", "ovr"],
            "break_ties": [False],
            "random_state": [None],
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
    