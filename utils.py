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
