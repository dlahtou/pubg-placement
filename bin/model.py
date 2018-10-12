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