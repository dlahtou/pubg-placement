import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import transformers
import model
from sklearn import metrics
from os.path import isdir, join
from os import mkdir
import pickle as pkl
import numpy as np

def main():
    out_folder = 'output'
    logging.basicConfig(level=logging.DEBUG)

    observations = extract()

    observations = transform(observations)

    observations, results = train(observations)

    load(observations, results, out_folder)

def extract():
    logging.info('Beginning Extract')

    observations = pd.read_csv('data/train.csv', nrows=500000) # TODO: remove nrows when testing on all data

    logging.info(f'Extract complete: {type(observations)} {observations.shape if isinstance(observations, pd.core.frame.DataFrame) else "NOT DATAFRAME"}')
 
    return observations


def transform(observations):
    # TODO: make aggregations for all features by groupId, as groupId determines final placement
    target_observations = observations.drop_duplicates(subset='groupId')[['groupId', 'winPlacePerc']].copy()
    
    observations.drop(columns=['matchId', 'Id', 'winPlacePerc'], inplace=True)

    logging.info('Adding teamMinima columns')
    team_observations = transformers.team_minima(observations)

    logging.info('Adding teamMaxima columns')
    team_observations = team_observations.merge(transformers.team_maxima(observations), how='left', on='groupId')

    logging.info('Adding teamMeans columns')
    team_observations = team_observations.merge(transformers.team_means(observations), how='left', on='groupId')

    logging.info('Adding teamSums columns')
    team_observations = team_observations.merge(transformers.team_sums(observations), how='left', on='groupId')

    logging.info('Adding back winPlacePerc column')
    team_observations = team_observations.merge(target_observations, how='left', on='groupId')

    logging.info('Adding teamSize column')
    team_observations = team_observations.merge(transformers.team_size_column(observations), how='left', on='groupId')

    logging.info(f'Transform complete')
    logging.info(f'Observations shape {team_observations.shape}')

    print(team_observations['teamSize'].unique())

    return team_observations

def train(observations):
    logging.info('Begin train')

    # edit this line to select different models for training
    selected_models = ["Ridge", "GBoost"]

    # looping vars
    model_functions = {"Ridge": model.ridge_model,
                        "GBoost": model.gboost_model}
    results = [] # will be a list of dictionaries of model results

    # format X, y; train-test split
    X_cols = [column for column in observations.columns if column != 'winPlacePerc']
    X = observations[X_cols]
    y = observations['winPlacePerc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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
        result_dict['MAE'] = metrics.mean_absolute_error(y_test, y_pred)
        result_dict['Scaled MAE'] = np.sum(np.abs(y_test - y_pred)*observations['teamSize'])/np.sum(observations['teamSize'])

        print(f'R squared value for {model_name}: {result_dict["r_squared"]}')
        print(f'MAE for {model_name}: {result_dict["MAE"]}')
        print(f'Scaled MAE for {model_name}: {result_dict["Scaled MAE"]}')

        results.append(result_dict)
    
    logging.info('Training complete')
    
    return observations, results

def load(observations, results, save_folder):
    logging.info(f'Saving results into {save_folder}')
    if not isdir(save_folder):
        mkdir(save_folder)

    with open(join(save_folder, 'observations.pkl'), 'wb') as open_file:
        pkl.dump(observations, open_file)
    with open(join(save_folder, 'results.pkl'), 'wb') as open_file:
        pkl.dump(results, open_file)
    
    logging.info('Save complete')

if __name__ == '__main__':
    main()