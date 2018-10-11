'''Contains functions which will add columns to the dataframe or permute existing columns.

All functions should accept a dataframe
All functions should return a dataframe'''

import pandas as pd

def team_max_killplace(observations):
    observations['teamMaxKillPlace'] = observations['killPlace'].groupby(observations['groupId']) \
                                                                .transform('max')

    return observations