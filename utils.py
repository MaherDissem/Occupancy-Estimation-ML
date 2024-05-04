import random
import numpy as np
import sklearn
# import torch
import mlflow


def set_seed(seed=0, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    sklearn.utils.check_random_state(seed)
    # if with_torch:
    #     torch.manual_seed(seed)
    # if with_cuda:
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False


def log_mlflow(experiment_name, run_name, params, metrics, artifact_path=None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.end_run()
    if artifact_path:
        mlflow.log_artifact(artifact_path)


def log_experiment(
    model,
    run_pca,
    run_grid_search,
    split_ratio,
    n_components,
    seed,
    acc,
    prec,
    rec,
    f1_score,
    roc,
):
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
