# Our best model so far has had a train set L1 of 0.0292
# NOTE: delete nrows parameter from pd.read_csv() for porting to kaggle. nrows provided in this code for quick testing

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
from catboost import CatBoostRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def main():
    logging.basicConfig(level=logging.DEBUG)

    observations = extract('train')

    observations = transform(observations)

    observations, results = train(observations)
    
    del observations
    
    test_set = pd.read_csv('../input/test_V2.csv', nrows=50000)
    
    logging.info(test_set.shape)
    logging.info(test_set.head())
    
    test_set = transform(test_set)
    
    logging.info(test_set.shape)
    
    output_by_groupId = predict(test_set, results, 'Catboost')
    
    load(output_by_groupId)

def ridge_model(X, y):
    model = Ridge()

    model.fit(X, y)

    return model

'''def lightgbm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    train_dataset = lgb.Dataset(X_train, y_train, silent=False)
    test_dataset = lgb.Dataset(X_test, y_test, silent=False)
    
    param = {'boosting_type': 'gbdt',
            'num_leaves': 120,
            'objective': 'regression_l2',
            'metric': 'mae',
            'max_depth': -1,
            'learning_rate': .05,
            'min_data_in_leaf': 200,
            'max_bin': 200}
    
    model = lgb.train(param, train_set=train_dataset, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=100, valid_sets=test_dataset)
    
    return model'''

def catboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    eval_set = (X_test, y_test)

    model = CatBoostRegressor(loss_function='RMSE',
                            custom_metric='MAE',
                            learning_rate=0.05,
                            use_best_model=True)
    
    '''loss_function='RMSE',
                            custom_metric='MAE',
                            max_depth=10,
                            use_best_model=True,
                            learning_rate=0.05'''

    model.fit(X_train, y_train, eval_set=eval_set)

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

def team_ranks(team_observations):
    team_observations.set_index('groupId', inplace=True)
    rank_observations = team_observations.groupby('matchId').rank(pct=True)
    rank_observations.columns += 'rank'
    
    rank_observations.reset_index(inplace=True)
    
    return rank_observations


def extract(csv_name):
    logging.info('Beginning Extract')

    observations = pd.read_csv(f'../input/{csv_name}_V2.csv', nrows=50000)

    logging.info(f'Extract complete: {type(observations)} {observations.shape if isinstance(observations, pd.core.frame.DataFrame) else "NOT DATAFRAME"}')
 
    return observations

def transform(observations):
    if 'winPlacePerc' in observations.columns:
        target_observations = observations.drop_duplicates(subset='groupId')[['groupId', 'winPlacePerc', 'maxPlace', 'numGroups', 'matchDuration']].copy()
        match_id_column = observations.drop_duplicates(subset='groupId')[['groupId', 'matchId']].copy()
        
        # gotta filter this bad boy
        observations = observations[observations['groupId'] != '12dfbede33f92b']
    
        observations.drop(columns=['matchId', 'Id', 'winPlacePerc', 'maxPlace', 'numGroups', 'matchDuration', 'matchType'], inplace=True)
    else:
        target_observations = observations.drop_duplicates(subset='groupId')[['groupId', 'maxPlace', 'numGroups', 'matchDuration']].copy()
        match_id_column = observations.drop_duplicates(subset='groupId')[['groupId', 'matchId']].copy()
    
        observations.drop(columns=['matchId', 'Id', 'maxPlace', 'numGroups', 'matchDuration', 'matchType'], inplace=True)
        
    logging.info('Adding teamMinima columns')
    team_observations = team_minima(observations)
    
    logging.info(team_observations.shape)

    logging.info('Adding teamMaxima columns')
    team_observations = team_observations.merge(team_maxima(observations), left_on='groupId', right_index=True)
    
    logging.info(team_observations.shape)
    logging.info(team_observations.head())

    logging.info('Adding teamMeans columns')
    team_observations = team_observations.merge(team_means(observations), left_on='groupId', right_index=True)
    
    logging.info(team_observations.shape)

    logging.info('Adding teamSums columns')
    team_observations = team_observations.merge(team_sums(observations), left_on='groupId', right_index=True)
    
    logging.info(team_observations.shape)
    
    logging.info('Adding back matchId column')
    team_observations = team_observations.merge(match_id_column, left_on='groupId', right_on='groupId')
    
    logging.info(team_observations.shape)
    
    logging.info('Adding match rank columns')
    team_observations = team_observations.merge(team_ranks(team_observations), left_on='groupId', right_on='groupId')
    
    logging.info(team_observations.shape)
    
    logging.info('Adding back info columns')
    team_observations = team_observations.merge(target_observations, left_on='groupId', right_on='groupId')
    
    logging.info(team_observations.shape)
    
    team_observations.set_index('groupId', inplace=True)

    logging.info('Adding teamSize column')
    team_observations['teamSize'] = observations.groupby(['groupId'])['assists'].count()

    team_observations.drop('matchId', axis=1, inplace=True)

    logging.info(f'Transform complete')
    logging.info(f'Observations shape {team_observations.shape}')

    logging.info(team_observations['teamSize'].unique())
    
    logging.info(team_observations[team_observations.isnull().any(axis=1)])

    return team_observations

def train(observations):
    logging.info('Begin train')

    # edit this line to select different models for training
    selected_models = ["Ridge", "Catboost"]

    # looping vars
    model_functions = {"Ridge": ridge_model,
                        "Catboost": catboost_model}
    results = [] # will be a list of dictionaries of model results

    # format X, y; train-test split
    X_cols = [column for column in observations.columns if column != 'winPlacePerc']
    X = observations[X_cols]

    X = scaler.fit_transform(X)

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
        
        X = scaler.transform(observations)
        
        observations['winPlacePerc'] = model['estimator'].predict(X)

        observations['winPlacePerc'] = observations.apply(lambda row: round_prediction_to_nearest_possible(row), axis=1)
        
        observations['winPlacePerc'].clip(0.0, 1.0, inplace=True)

        observations.reset_index(inplace=True)
        
        output_by_groupId = observations[['groupId', 'winPlacePerc']]
        
    return output_by_groupId
    
def load(output_by_groupId):
    observations = extract('test')
    
    observations = observations.merge(output_by_groupId, how="left", on="groupId")
    
    output = observations[['Id', 'winPlacePerc']]
    
    output.to_csv('catboost_submission.csv', index=False)
    

if __name__ == '__main__':
    main()