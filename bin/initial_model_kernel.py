# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
import logging
from sklearn import metrics
from os.path import isdir, join
from os import listdir
import pickle as pkl
import gc

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

def ridge_model(X, y):
    model = Ridge()

    model.fit(X, y)

    return model

def gboost_model(X, y):
    model = GradientBoostingRegressor()

    model.fit(X, y)

    return model

def team_minima(observations):
    team_observations = observations.groupby(observations['groupId']).min()
    team_observations.columns += 'min'

    return team_observations

def team_maxima(observations):
    team_observations = observations.groupby(observations['groupId']).max()
    team_observations.columns += 'max'

    return team_observations

def team_means(observations):
    team_observations = observations.groupby(observations['groupId']).mean()
    team_observations.columns += 'mean'

    return team_observations

def team_sums(observations):
    team_observations = observations.groupby(observations['groupId']).sum()
    team_observations.columns += 'sum'

    return team_observations

def main():
    logging.basicConfig(level=logging.DEBUG)

    observations = extract('train')

    observations = transform(observations)

    observations, results = train(observations)
    
    del observations
    
    test_set = pd.read_csv('../input/test_V2.csv')
    
    logging.info(test_set.shape)
    logging.info(test_set.head())
    
    test_set = transform(test_set)
    
    logging.info(test_set.shape)
    
    output_by_groupId = predict(test_set, results, 'GBoost')
    
    load(output_by_groupId)

def extract(csv_name):
    logging.info('Beginning Extract')

    observations = pd.read_csv(f'../input/{csv_name}_V2.csv')

    logging.info(f'Extract complete: {type(observations)} {observations.shape if isinstance(observations, pd.core.frame.DataFrame) else "NOT DATAFRAME"}')
 
    return observations


def transform(observations):
    if 'winPlacePerc' in observations.columns:
        target_observations = observations.drop_duplicates(subset='groupId')[['groupId', 'winPlacePerc', 'maxPlace', 'numGroups', 'matchDuration']].copy()
        
        # gotta filter this bad boy
        observations = observations[observations['groupId'] != '12dfbede33f92b']
    
        observations.drop(columns=['matchId', 'Id', 'winPlacePerc', 'maxPlace', 'numGroups', 'matchDuration', 'matchType'], inplace=True)
    else:
        target_observations = observations.drop_duplicates(subset='groupId')[['groupId', 'maxPlace', 'numGroups', 'matchDuration']].copy()
    
        observations.drop(columns=['matchId', 'Id', 'maxPlace', 'numGroups', 'matchDuration', 'matchType'], inplace=True)
        
    logging.info('Adding teamMinima columns')
    team_observations = team_minima(observations)
    
    logging.info(team_observations.shape)

    logging.info('Adding teamMaxima columns')
    team_observations = team_observations.merge(team_maxima(observations), how='left', on='groupId')
    
    logging.info(team_observations.shape)
    logging.info(team_observations.head())

    logging.info('Adding teamMeans columns')
    team_observations = team_observations.merge(team_means(observations), how='left', on='groupId')
    
    logging.info(team_observations.shape)

    logging.info('Adding teamSums columns')
    team_observations = team_observations.merge(team_sums(observations), how='left', on='groupId')
    
    logging.info(team_observations.shape)
    
    logging.info('Adding back info columns')
    team_observations = team_observations.merge(target_observations, how='left', on='groupId')
    
    logging.info(team_observations.shape)
    
    team_observations.set_index('groupId', inplace=True)

    logging.info('Adding teamSize column')
    team_observations['teamSize'] = observations.groupby(['groupId'])['assists'].count()

    logging.info(f'Transform complete')
    logging.info(f'Observations shape {team_observations.shape}')

    logging.info(team_observations['teamSize'].unique())
    
    logging.info(team_observations[team_observations.isnull().any(axis=1)])

    return team_observations

def train(observations):
    logging.info('Begin train')

    # edit this line to select different models for training
    selected_models = ["GBoost"]

    # looping vars
    model_functions = {"Ridge": ridge_model,
                        "GBoost": gboost_model}
    results = [] # will be a list of dictionaries of model results

    # format X, y; train-test split
    X_cols = [column for column in observations.columns if column != 'winPlacePerc']
    X = observations[X_cols]
    y = observations['winPlacePerc']

    # run every model and append a dictionary of outcomes to results list
    for model_name in selected_models:
        logging.info(f'Training {model_name} model')
        estimator = model_functions[model_name](X, y)

        y_pred = estimator.predict(X)

        result_dict = dict()
        result_dict['model_label'] = model_name
        result_dict['estimator'] = estimator
        result_dict['r_squared'] = metrics.r2_score(y, y_pred)
        result_dict['MSE'] = metrics.mean_squared_error(y, y_pred)
        result_dict['MAE'] = metrics.mean_absolute_error(y, y_pred)
        result_dict['Scaled MAE'] = np.sum(np.abs(y - y_pred)*observations['teamSize'])/np.sum(observations['teamSize'])

        logging.info(f'R squared value for {model_name}: {result_dict["r_squared"]}')
        logging.info(f'MAE for {model_name}: {result_dict["MAE"]}')
        logging.info(f'Scaled MAE for {model_name}: {result_dict["Scaled MAE"]}')

        results.append(result_dict)
    
    logging.info('Training complete')
    
    with open('models.pkl', 'wb') as open_file:
        pkl.dump(results, open_file)
    
    return observations, results
    
def round_prediction_to_nearest_possible(row):
    maxPlace = row['maxPlace']
    pred = row['winPlacePerc']
    
    step = 1.0 / (maxPlace - 1.0)
    
    pred_under = step * (pred // step)
    pred_over = step * ((pred // step) + 1)
    
    return pred_under if abs(pred_under - pred) < abs(pred_over - pred) else pred_over
    
def predict(observations, results, model_name):
    for model in results:
        if model['model_label'] != model_name:
            continue
        
        observations['winPlacePerc'] = model['estimator'].predict(observations)
        
        observations['winPlacePerc'] = observations.apply(lambda row: round_prediction_to_nearest_possible(row), axis=1)

        observations.reset_index(inplace=True)
        
        output_by_groupId = observations[['groupId', 'winPlacePerc']]
        
    return output_by_groupId
    
def load(output_by_groupId):
    observations = extract('test')
    
    observations = observations.merge(output_by_groupId, how="left", on="groupId")
    
    output = observations[['Id', 'winPlacePerc']]
    
    output.to_csv('second_submission.csv', index=False)
    

if __name__ == '__main__':
    main()