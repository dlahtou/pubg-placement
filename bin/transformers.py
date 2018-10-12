'''Contains functions which will add columns to the dataframe or permute existing columns.

All functions should accept a dataframe
All functions should return a dataframe'''

import pandas as pd

def team_min_killplace(observations):
    observations['teamMinKillPlace'] = observations['killPlace'].groupby(observations['groupId']) \
                                                                .transform('min')

    return observations

def team_size(observations):
    observations['teamSize'] = observations.groupby(observations['groupId'])['assists'] \
                                            .transform('count')

    return observations

def team_assists(observations):
    observations['teamAssists'] = observations['assists'].groupby(observations['groupId']) \
                                                        .transform('sum')
    
    return observations

def team_max_walkdistance(observations):
    observations['teamMaxWalkDistance'] = observations['walkDistance'].groupby(observations['groupId']) \
                                                        .transform('max')

    return observations

def team_size_column(observations):
    teamsize_observations = observations[['groupId', 'assists']].copy()
    teamsize_observations['teamSize'] = teamsize_observations.groupby(teamsize_observations['groupId'])['assists'] \
                                        .transform('count')
    
    return teamsize_observations.drop(columns=['assists'])

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