# Occupancy-Estimation-ML

Course project for the course "INSE 6220 - Advanced Statistical Approaches to Quality", taken at Concordia University during the Winter 2024 term.

This project aims to perform and compare simple Machine Learning algorithms and binary classification models on the occupancy estimation task for smart buildings. 
To this end, we employ the dataset from [here](https://g2elab.grenoble-inp.fr/fr/plateformes/predis-mhi) which consist of various sensors records captured within an office space at the Grenoble Institute of Technology. This dataset is available upon request.

This repository is organized as follows:

- `dataset/sql2csv.ipynb` load the raw SQL dataset and computes a new dataset to be saved in `data/processed`. 
This involves, discarding irrelevant features, resampling observations using a larger sampling rate as they are not recorded for the same timestamps and some additional simple pre-processing.

- `dataset/EDA.ipynb` provides an exploratory data analysis and is used to generate the plots displayed in the report.

- `main.py` will run PCA (`PCA.py`), train the ML classifiers (models are defined in `models/`), perform hyperparameter optimization through grid-search, evaluate the model on the test data and save results to an MLflow server (to be started with `mlflow ui -p 8080`).

- `run_experiments.py` will run and log multiple experiments for different models and parameters.

To replicate our results, run the following:

``````
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

python run_experiments.py
python -m mlflow ui -p 8080
``````
