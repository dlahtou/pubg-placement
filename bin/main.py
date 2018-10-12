import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import transformers
import model
from sklearn import metrics
from os.path import isdir, join
from os import mkdir
import pickle as pkl

def main():
    out_folder = 'output'
    logging.basicConfig(level=logging.DEBUG)

    observations = extract()

    observations = transform(observations)

    observations, results = train(observations)

    load(observations, results, out_folder)

def extract():
    logging.info('Beginning Extract')

    observations = pd.read_csv('data/train.csv', nrows=1000000) # TODO: remove nrows when testing on all data

    logging.info(f'Extract complete: {type(observations)} {observations.shape if isinstance(observations, pd.core.frame.DataFrame) else "NOT DATAFRAME"}')
 
    return observations


def transform(observations):
    # TODO: create synthetic features; data cleaning not necessary

    logging.info('Adding teamMaxKillPlace column')
    observations = transformers.team_max_killplace(observations)

    logging.info(f'Transform complete')

    return observations

def train(observations):
    logging.info('Begin train')

    # edit this line to select different models for training
    selected_models = ["Ridge"]

    # looping vars
    model_functions = {"Ridge": model.ridge_model}
    results = [] # will be a list of dictionaries of model results

    # format X, y; train-test split
    X_cols = [column for column in observations.columns if column != 'winPlacePerc']
    X = observations[X_cols]
    y = observations['winPlacePerc']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # run every model and append a dictionary of outcomes to results list
    for model_name in selected_models:
        logging.info(f'Training {model_name} model')
        estimator = model_functions[model_name](X_train, y_train)

        y_pred = estimator.predict(X_test)

        result_dict = dict()
        result_dict['model_label'] = model_name
        result_dict['estimator'] = estimator
        result_dict['r_squared'] = metrics.r2_score(y_test, y_pred)
        result_dict['MSE'] = metrics.mean_squared_error(y_test, y_pred)

        results.append(result_dict)
    
    logging.info('Training complete')
    
    return observations, results

def load(observations, results, save_folder):
    logging.info(f'Saving results into {save_folder}')
    if not isdir(save_folder):
        mkdir(save_folder)

    with open(join(save_folder, 'observations.pkl')) as open_file:
        pkl.dump(observations, open_file)
    with open(join(save_folder, 'results.pkl')) as open_file:
        pkl.dump(results, open_file)
    
    logging.info('Save complete')

if __name__ == '__main__':
    main()