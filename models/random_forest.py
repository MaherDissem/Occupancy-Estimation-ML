import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from models.base_classifier import BaseClassifier

class RandomForest(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
        )

        self.param_grid = {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
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
    