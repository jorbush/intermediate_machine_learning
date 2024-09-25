import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


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

def missing_values():
    # Load the data
    data = pd.read_csv('./input/melb_data.csv')
    # Select target
    y = data.Price
    # To keep things simple, we'll use only numerical predictors
    melb_predictors = data.drop(['Price'], axis=1)
    X = melb_predictors.select_dtypes(exclude=['object'])
    # Divide data into training and validation subsets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8,
                                                          test_size=0.2, random_state=0)

    # Function for comparing different approaches
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    # Get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]

    # Drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print("MAE from Approach 1 (Drop columns with missing values):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print("MAE from Approach 2 (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

    # Make copy to avoid changing original data (when imputing)
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    print("MAE from Approach 3 (An Extension to Imputation):")
    print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

    # Shape of training data (num_rows, num_columns)
    print(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

def missing_values_exercise():
    # Read the data
    X_full = pd.read_csv('./input/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./input/test.csv', index_col='Id')
    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)
    # To keep things simple, we'll use only numerical predictors
    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])
    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8,
                                                          test_size=0.2, random_state=0)
    print(X_train.head())

    # Shape of training data (num_rows, num_columns)
    print(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # Fill in the line below: How many rows are in the training data?
    num_rows = X_train.shape[0]
    print(num_rows)

    # Fill in the line below: How many columns in the training data
    # have missing values?
    num_cols_with_missing = len([col for col in X_train.columns
                         if X_train[col].isnull().any()])
    print(num_cols_with_missing)

    # Fill in the line below: How many missing entries are contained in
    # all of the training data?
    tot_missing = X_train.isnull().sum().sum()
    print(tot_missing)

    # Function for comparing different approaches
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    # Fill in the line below: get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]

    # Fill in the lines below: drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print("MAE (Drop columns with missing values):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

    # Fill in the lines below: imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Fill in the lines below: imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print("MAE (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

    # Preprocessed training and validation features
    my_imputer = SimpleImputer()
    final_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    final_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Define and fit model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(final_X_train, y_train)

    # Get validation predictions and MAE
    preds_valid = model.predict(final_X_valid)
    print("MAE (Your approach):")
    print(mean_absolute_error(y_valid, preds_valid))

    # Fill in the line below: preprocess test data
    final_X_test = pd.DataFrame(my_imputer.transform(X_test))

    # Fill in the line below: get test predictions
    preds_test = model.predict(final_X_test)

def categorical_variables():
    # Read the data
    data = pd.read_csv('./input/melb_data.csv')

    # Separate target from predictors
    y = data.Price
    X = data.drop(['Price'], axis=1)

    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)

    # Drop columns with missing values (simplest approach)
    cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
    X_train_full.drop(cols_with_missing, axis=1, inplace=True)
    X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                            X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = low_cardinality_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    print(X_train.head())

    # Get list of categorical variables
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)

    print("Categorical variables:")
    print(object_cols)

    # Function for comparing different approaches
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    print("MAE from Approach 1 (Drop categorical variables):")
    print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

    # Make copy to avoid changing original data
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()

    # Apply ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
    label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

    print("MAE from Approach 2 (Ordinal Encoding):")
    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore')
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]).toarray())
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]).toarray())

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    # Ensure all columns have string type
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)

    print("MAE from Approach 3 (One-Hot Encoding):")
    print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

if __name__ == '__main__':
    # introduction_exercise()
    # missing_values()
    # missing_values_exercise()
    categorical_variables()
