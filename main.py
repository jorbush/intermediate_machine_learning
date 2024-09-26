import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

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

def categorical_variables_exercise():
    # Read the data
    X = pd.read_csv('./input/train.csv', index_col='Id')
    X_test = pd.read_csv('./input/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # To keep things simple, we'll drop columns with missing values
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8, test_size=0.2,
                                                          random_state=0)
    print(X_train.head())

    # function for comparing different approaches
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    # Fill in the lines below: drop columns in training and validation data
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])
    print("MAE from Approach 1 (Drop categorical variables):")
    print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
    print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
    print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

    # Categorical columns in the training data
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columns that can be safely ordinal encoded
    good_label_cols = [col for col in object_cols if
                       set(X_valid[col]).issubset(set(X_train[col]))]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    print('Categorical columns that will be ordinal encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

    # Drop categorical columns that will not be encoded
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

    # Apply ordinal encoder
    ordinal_encoder = OrdinalEncoder()
    label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
    label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    # Print number of unique entries by column, in ascending order
    sorted(d.items(), key=lambda x: x[1])

    # Fill in the line below: How many categorical variables in the training data
    # have cardinality greater than 10?
    high_cardinality_numcols = len([cname for cname in X_train.columns if X_train[cname].nunique() > 10 and
                                X_train[cname].dtype == "object"])

    # Fill in the line below: How many columns are needed to one-hot encode the
    # 'Neighborhood' variable in the training data?
    num_cols_neighborhood = X_train['Neighborhood'].nunique()

    # Fill in the line below: How many entries are added to the dataset by
    # replacing the column with a one-hot encoding?
    OH_entries_added = 1e4*100 - 1e4

    # Fill in the line below: How many entries are added to the dataset by
    # replacing the column with an ordinal encoding?
    label_entries_added = 0

    # Columns that will be one-hot encoded
    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

    # Columns that will be dropped from the dataset
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
    print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

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

def pipelines():
    # Read the data
    data = pd.read_csv('./input/melb_data.csv')
    # Separate target from predictors
    y = data.Price
    X = data.drop(['Price'], axis=1)
    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]
    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    print(X_train.head())

    ''' Step 1: Define Preprocessing Steps '''

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    '''Step 2: Define the Model'''

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    '''Step 3: Create and Evaluate the Pipeline'''

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('MAE:', score)

def pipelines_exercise():
    # Read the data
    X_full = pd.read_csv('./input/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./input/test.csv', index_col='Id')
    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)
    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                                    train_size=0.8, test_size=0.2,
                                                                    random_state=0)
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train_full.columns if
                        X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]
    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if
                      X_train_full[cname].dtype in ['int64', 'float64']]
    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()
    print(X_train.head())

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    # Define model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                          ])
    # Preprocessing of training data, fit model
    clf.fit(X_train, y_train)
    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_valid)
    print('MAE:', mean_absolute_error(y_valid, preds))

    # Preprocessing of test data, fit model
    preds_test = clf.predict(X_test)

def cross_validation():
    # In cross-validation, we run our modeling process on different
    # subsets of the data to get multiple measures of model quality.
    # For small datasets

    # Read the data
    data = pd.read_csv('./input/melb_data.csv')
    # Select subset of predictors
    cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
    X = data[cols_to_use]
    # Select target
    y = data.Price

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(
                                      n_estimators=50,random_state=0))])
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')

    print("MAE scores:\n", scores)

    print("Average MAE score (across experiments):")
    print(scores.mean())

def cross_validation_exercise():
    # Read the data
    train_data = pd.read_csv('./input/train.csv', index_col='Id')
    test_data = pd.read_csv('./input/test.csv', index_col='Id')
    # Remove rows with missing target, separate target from predictors
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data.SalePrice
    train_data.drop(['SalePrice'], axis=1, inplace=True)
    # Select numeric columns only
    numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
    X = train_data[numeric_cols].copy()
    X_test = test_data[numeric_cols].copy()

    print(X.head())
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ])

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')

    print("Average MAE score:", scores.mean())

    def get_score(n_estimators):
        """Return the average MAE over 3 CV folds of random forest model.

        Keyword argument:
        n_estimators -- the number of trees in the forest
        """
        # Replace this body with your own code
        my_pipeline = Pipeline(steps=[
            ('preprocessor', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators, random_state=0))
        ])

        # Multiply by -1 since sklearn calculates *negative* MAE
        scores = -1 * cross_val_score(my_pipeline, X, y,
                                      cv=3,
                                      scoring='neg_mean_absolute_error')
        return scores.mean()
    results = {x: get_score(x) for x in range(50, 401, 50)}
    print(results)
    # plt.plot(list(results.keys()), list(results.values()))
    # plt.show()

    n_estimators_best = min(results, key=results.get)
    print(n_estimators_best)


def xgboost():
    # optimize models with gradient boosting
    # alternative to random forests

    # Gradient boosting is a method that goes through cycles to
    # iteratively add models into an ensemble.

    # Read the data
    data = pd.read_csv('./input/melb_data.csv')
    # Select subset of predictors
    cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
    X = data[cols_to_use]
    # Select target
    y = data.Price
    # Separate data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    my_model = XGBRegressor()
    my_model.fit(X_train, y_train)

    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

    my_model = XGBRegressor(n_estimators=500)
    my_model.fit(X_train, y_train)

    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

    my_model = XGBRegressor(n_estimators=500, early_stopping_rounds=5)
    my_model.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)

    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

    my_model = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, learning_rate=0.05)
    my_model.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)

    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

    my_model = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, learning_rate=0.05, n_jobs=8)
    my_model.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)

    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

def xgboost_exercise():
    # Read the data
    X = pd.read_csv('./input/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./input/test.csv', index_col='Id')
    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)
    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                            X_train_full[cname].dtype == "object"]
    # Select numeric columns
    numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
    # Keep selected columns only
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()
    # One-hot encode the data (to shorten the code, we use pandas)
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    my_model_1 = XGBRegressor(random_state=0)
    my_model_1.fit(X_train, y_train)
    predictions_1 = my_model_1.predict(X_valid)
    print("Mean Absolute Error model_1: " + str(mean_absolute_error(predictions_1, y_valid)))

    my_model_2 = XGBRegressor(n_estimators=500, early_stopping_rounds=5, learning_rate=0.01, n_jobs=8)
    my_model_2.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)

    predictions_2 = my_model_2.predict(X_valid)
    print("Mean Absolute Error model_2: " + str(mean_absolute_error(predictions_2, y_valid)))

    my_model_3 = XGBRegressor(n_estimators=1000, early_stopping_rounds=10, learning_rate=0.9, n_jobs=1)
    my_model_3.fit(X_train, y_train,
                   eval_set=[(X_valid, y_valid)],
                   verbose=False)

    predictions_3 = my_model_3.predict(X_valid)
    print("Mean Absolute Error model_3: " + str(mean_absolute_error(predictions_3, y_valid)))

if __name__ == '__main__':
    # introduction_exercise()
    # missing_values()
    # missing_values_exercise()
    # categorical_variables()
    # categorical_variables_exercise()
    # pipelines()
    # pipelines_exercise()
    # cross_validation()
    # cross_validation_exercise()
    # xgboost()
    xgboost_exercise()