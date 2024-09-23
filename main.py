import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def introduction_exercise():
    # Read the data
    X_full = pd.read_csv('./input/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./input/test.csv', index_col='Id')

    # Obtain target and predictors
    y = X_full.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = X_full[features].copy()
    X_test = X_test_full[features].copy()

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    print(X_train.head())

    # Define the models
    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

    models = [model_1, model_2, model_3, model_4, model_5]

    # Function for comparing different models
    def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return mean_absolute_error(y_v, preds)

    best_model, best_mae = model_1, float('inf')
    for i in range(0, len(models)):
        mae = score_model(models[i])
        model_id = i + 1
        if mae < best_mae:
            best_model = models[i]
            best_mae = mae
        print("Model %d MAE: %d" % (model_id, mae))

    # Fill in the best model
    print(best_model)

    # Define a model
    my_model = RandomForestRegressor(n_estimators=120, criterion='absolute_error', random_state=0)

    # Fit the model to the training data
    my_model.fit(X, y)

    # Generate test predictions
    preds_test = my_model.predict(X_test)
    models.append(my_model)

    for i in range(0, len(models)):
        mae = score_model(models[i])
        model_id = i + 1
        print("Model %d MAE: %d" % (model_id, mae))

if __name__ == '__main__':
    introduction_exercise()

