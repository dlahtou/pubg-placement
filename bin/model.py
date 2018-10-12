from sklearn.linear_model import Ridge

def ridge_model(X, y):
    model = Ridge()

    model.fit(X, y)

    return model