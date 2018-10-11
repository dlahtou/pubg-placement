import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import transformers

def main():
    logging.basicConfig(level=logging.DEBUG)

    observations = extract()

    observations = transform(observations)

    observations, results = train(observations)

    load(observations, results)

def extract():
    logging.info('Beginning Extract')

    observations = pd.read_csv('data/train.csv', nrows=1000000) # TODO: remove nrows when testing on all data

    logging.info(f'Extract complete: {type(observations)} {observations.shape if isinstance(observations, pd.core.frame.DataFrame) else "NOT DATAFRAME"}')
 
    return observations


def transform(observations):
    # TODO: create synthetic features; data cleaning not necessary

    logging.info('Adding teamMaxKillPlace column')
    observations = transformers.team_max_killplace(observations)

    print(observations.info())
    observations.sort_values(by='groupId', inplace=True)
    print(observations.head())

    logging.info(f'Transform complete')

    return observations

def train(observations):
    return 1, 2

def load(observations, results):
    pass

if __name__ == '__main__':
    main()