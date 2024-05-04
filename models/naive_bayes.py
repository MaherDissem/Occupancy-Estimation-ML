import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from models.base_classifier import BaseClassifier


class NaiveBayes(BaseClassifier):
    def __init__(self) -> None:
        self.model = GaussianNB()

        self.param_grid = {
            "priors": [None],
            "var_smoothing": [1e-9],
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