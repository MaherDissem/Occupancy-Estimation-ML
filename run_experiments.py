import pandas as pd

from models.naive_bayes import NaiveBayes
from models.svm import SVMclassifier
from models.knn_classifier import KNNClassifier
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from main import experiment
from utils import log_experiment


split_ratio = 0.2
run_grid_search = True
run_pca = True
n_components = 2  # originally, the dataset has 13 features
seed = 42
csv_path = "data/processed/h358-2015.csv"
data = pd.read_csv(csv_path, index_col=0)


for model in [
    NaiveBayes(),
    SVMclassifier(),
    KNNClassifier(),
    DecisionTree(),
    RandomForest(),
]:

    # run with PCA
    run_pca = True
    acc, prec, rec, f1_score, cm, roc, report = experiment(
        data, model, split_ratio, run_pca, run_grid_search, n_components, seed
    )
    log_experiment(
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
    )

    # run without PCA
    run_pca = False
    acc, prec, rec, f1_score, cm, roc, report = experiment(
        data, model, split_ratio, run_pca, run_grid_search, n_components, seed
    )
    log_experiment(
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
    )
