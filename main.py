from typing import List, Tuple
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

from PCA import perform_pca
from models.base_classifier import BaseClassifier
from models.decision_tree import DecisionTree
from metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    roc_auc,
    classification_report,
)
from utils import set_seed
from utils import log_mlflow

warnings.simplefilter(action="ignore", category=FutureWarning)


def experiment(
    data: pd.DataFrame,
    model: BaseClassifier,
    split_ratio: float = 0.2,
    run_pca: bool = False,
    run_grid_search: bool = False,
    n_components: int = 3,
    seed: int = 42,
) -> Tuple[float, float, float, float, List[List[int]], float, str]:
    set_seed(seed)

    # Process data
    X = data.drop(columns=["OCCUPANCY_14"])
    y = data["OCCUPANCY_14"]
    if run_pca:
        X = perform_pca(X, n_components)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=seed
    )

    # Train and predict
    model.fit(X_train, y_train, run_grid_search)
    y_pred = model.predict(X_test)

    # Print optimal parameters
    print(model.get_params()) # TODO log this to mlflow

    # Evaluate model
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1_score = f1(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, prec, rec, f1_score, cm, roc, report


if __name__ == "__main__":
    csv_path = "data/processed/h358-2015.csv"
    data = pd.read_csv(csv_path, index_col=0)

    model = DecisionTree()
    split_ratio = 0.2
    run_grid_search = True
    run_pca = True
    n_components = 3  # originally, the dataset has 15 features
    seed = 42

    acc, prec, rec, f1_score, cm, roc, report = experiment(
        data, model, split_ratio, run_pca, run_grid_search, n_components, seed
    )

    log_mlflow(
        experiment_name="Occupancy Estimation",
        run_name=f"{model.__class__.__name__} - PCA: {run_pca} - Grid Search: {run_grid_search}",
        params={
            "split_ratio": split_ratio,
            "run_grid_search": run_grid_search,
            "run_pca": run_pca,
            "n_components": n_components,
            "seed": seed,
        },
        metrics={
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1_score,
            "roc_auc": roc,
        },
    )

    print(
        f"> Accuracy: {acc:.2f}\n"
        f"> Precision: {prec:.2f}\n"
        f"> Recall: {rec:.2f}\n"
        f"> F1 Score: {f1_score:.2f}\n"
        f"> Confusion Matrix: {cm}\n"
        f"> ROC AUC: {roc:.2f}\n"
        f"> Classification Report: {report}\n"
    )
