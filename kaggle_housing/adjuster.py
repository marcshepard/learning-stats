"""
Experiment with the concept of an "adjuster" for the housing model.
Idea is:
* First fit a linear model, starting with just numeric features and pruning out the ones not useful,
  and adding a handful of categorical features if useful.
* Next, fit a tree-based model (e.g. random forest) to the residuals of the linear model, to capture
  the non-linearities.
"""

# Ignore certain pyline warnings
# pylint: disable=invalid-name, line-too-long, import-error

# Imports
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score

# For reproduceability
np.random.seed(12)

TRACE_DBG = False   # Controls debug output messages

def trace_dbg(*args, **kwargs):
    """ conditionally print """
    if TRACE_DBG:
        print (args, kwargs)

def print_metrics(yhat_train, y_train, yhat_val, y_val):
    """ Print metrics for the model, where y_true and y_pred log-transformed numpy arrays """
    trace_dbg (f"Val R^2 : {r2_score(yhat_val, y_val):.4f}\t Train R^2 : {r2_score(yhat_train, y_train):.4f}")
    print (f"Val RMSE: {np.sqrt(mean_squared_error(yhat_val, y_val)):.4f}\t Train RMSE: {np.sqrt(mean_squared_error(yhat_train, y_train)):.4f}")
    yhat_val = np.exp(yhat_val)
    y_val = np.exp(y_val)
    trace_dbg (f"Val MAE of unnormalized prices: {mean_absolute_error(yhat_val, y_val):.0f}")
    trace_dbg (f"Val MPE of unnormalized prices: {mean_absolute_percentage_error(yhat_val, y_val):.2%}")

def print_feature_importances(model, df, label, dropped_cols=()):
    """ Print feature importances for the model, where df is a pandas dataframe and label is the target variable
    The model is split into train and validation sets, trained, then feature importance is calculated by checking the RMSE
    for each column when it is set to it's mean value (vs a baseline RMSE when all column values are used as is)
    """
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(df.drop(label, axis=1), df[label], test_size=.2, random_state=12)
    model.fit(X_train, y_train)

    # Calculate feature importance by checking the RMSE for each column when it is set to it's mean value
    importance = {}
    useless = []
    X = X_val.copy()
    yhat = model.predict(X)
    baseline_rmse = np.sqrt(mean_squared_error(y_val, yhat))
    for col in X_val.columns:
        X = X_val.copy()
        # For numeric columns, set the value to the mean of the column
        if np.issubdtype(X[col].dtype, np.number):
            X[col] = X[col].mean()
        # For categorical columns, set the value to the most frequent value in the column
        else:
            X[col] = X[col].mode()[0]
        yhat = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_val, yhat))
        if rmse <= baseline_rmse:
            useless.append(col)
        else:
            importance[col] = rmse - baseline_rmse

    useless = set(useless) - set(dropped_cols)
    print ("Useless columns: ", sorted(useless))
    print ("Columns that reduce the RMSE:")
    for col, rmse in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        if col in dropped_cols:
            col += " (DROPPED!)"
        print (f"{col}: {rmse:.4f}")

class AveragePredictor(BaseEstimator):
    """ Average the prediction of serveral models """
    def __init__(self, models):
        self.models = models

    def fit(self, X, y=None):
        """ Fit all  models """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """ Average the predictions of all models """
        yhat = np.mean([model.predict(X) for model in self.models], axis=0)
        return yhat

class TypeSelector(TransformerMixin):
    """ Pipeline componenet to select columns of a particular type; needed for column transformers """
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):   # pylint: disable=unused-argument
        """ Fit the selector; nothing to do"""
        return self

    def transform(self, X):
        """ Transform the data by selecting only columns of the specified type """
        return X.select_dtypes(include=[self.dtype])

class DataCleaner(TransformerMixin):
    """ Clean up the input data based on docs to remove NaNs, adjust data types, etc """
    def fit(self, X, y=None):   # pylint: disable=unused-argument
        """ Nothing to fit """
        return self

    def transform(self, df):
        """ Preprocess the data """
        df = df.copy()

        # Drop useless Id column (it is not a predictor)
        df.drop("Id", axis=1, inplace=True)

        # Covert MSSubClass from numeric to categorical, since the data is categorical
        df['MSSubClass'] = df['MSSubClass'].astype(str)

        # Fill in missing numeric values with the correct detaults
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
        cols = df.select_dtypes(include=[np.number]).columns        # All others get 0's
        null_cols = [col for col in cols if df[col].isnull().sum() > 0]
        for col in null_cols:
            trace_dbg (f"Column {col} has {df[col].isnull().sum()/df.shape[0]:.2%} nulls, inputing to 0")
            df[col].fillna(0, inplace=True)

        # Fill in missing categorical values with "NA" if they are missing because the feature doesn't apply
        cols = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "ExterQual", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                "FireplaceQu", "Fence", "KitchenQual", "PoolQC", "MiscFeature"]
        for col in cols:
            if  df[col].isnull().sum() > 0:
                trace_dbg (f"Column {col} has {df[col].isnull().sum()/df.shape[0]:.2%} nulls, inputing to 'NA'")
                df[col].fillna('NA', inplace=True)

        # Just guessing the right value to impute here...
        df["Electrical"].fillna('Mix', inplace=True)

        # While I've not trained on the test data, the code below shows it has nulls not found in the training data, so we'll have to guess how to deal with them
        df["Exterior2nd"].fillna('None', inplace=True)      # Not currently in final model
        df["Utilities"].fillna('None', inplace=True)        # Not currently in final model
        df["SaleType"].fillna('Unknown', inplace=True)      # Not currently used in final model
        df["MSZoning"].fillna('Unknown', inplace=True)      # Currently in final model (linear) - can it be removed or only used conditionally?
        df["Functional"].fillna('Typ', inplace=True)        # Currently in final model (linear) - can it be removed or only used conditionally?
        df["Exterior1st"].fillna('Unknown', inplace=True)   # Currently in final model (linear) - can it be removed or only used conditionally?

        null_cols = set(col for col in df.columns if df[col].isnull().sum() > 0)
        for col in null_cols:
            print (f"Warning: column {col} has {df[col].isnull().sum()/df.shape[0]:.2%} nulls")

        return df

class LinearModelTransformer(TransformerMixin):
    """ Prepare the data for a linear model """

    def fit (self, X, y=None):  # pylint: disable=unused-argument
        """ Nothing to fit """
        return self

    def transform(self, df):
        """ Transform the data in prep for a linear model """

        # Drop categorical columns after one-hot encoding or labele encoding a select few
        # Adjust the skewness of a few numeric columns
        # After this is run, we'll look at feature importance and drop the ones that don't help

        # Unskew a few numeric features
        for cols in ["GrLivArea", "LotArea", "1stFlrSF"]:
            skew = df[cols].skew()
            df[cols] = np.log(df[cols])
            trace_dbg (f"Unskewing {cols} from {skew:.2f} to {df[cols].skew():.2f}")

        # Total baths is a better predictor than individual bath counts
        df["TotalBaths"] = df["FullBath"] + .5* df["HalfBath"] + df["BsmtFullBath"] + .5 * df["BsmtHalfBath"]
        df.drop(["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], axis=1, inplace=True)

        # Label encoding...
        for col_name in ["ExterQual", "KitchenQual", "GarageQual", "BsmtQual", "GarageCond", "HeatingQC", "FireplaceQu"]:
            df[col_name] = df[col_name].map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'NA':0}, na_action='ignore')
            assert df[col_name].isnull().sum() == 0, f"{col_name} still has {int(df[col_name].isnull().sum()/df.shape[0] * 100)}% nulls)"
            assert np.issubdtype(df[col_name], np.number), f"{col_name} is not a number"
        df["Electrical"] = df["Electrical"].map({'SBrkr':4, 'FuseA':3, 'FuseF':2, 'FuseP':1, 'Mix':0})

        # Onehot encoding
        df["KitchenAbvGr"] = df["KitchenAbvGr"].apply (lambda x: 1 if x == 1 else 0)
        df["CentralAir"] = df.CentralAir.apply(lambda x: 1 if x == 'Y' else 0)              # Central air adds $500 (not much, but it helps)
        df["ZoneCommercial"] = df["MSZoning"].apply (lambda x: 1 if x == 'C (all)' else 0)

        # Drop columns that have have been shown to not help in order to improves linear model performance
        df.drop(self.drop_cols, axis=1, inplace=True)

        return df

    @property
    def drop_cols(self):
        """ Return a list of columns that get dropped because they help neither linear model """
        return ['3SsnPorch', 'Alley', 'BedroomAbvGr', 'BldgType', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Condition2',
                'Electrical', 'ExterCond', 'Exterior2nd', 'ExterQual', 'Fence',
                'Fireplaces', 'GarageArea', 'GarageCond', 'GarageType', 'GarageYrBlt', 'Heating', 'HouseStyle', 'LandSlope',
                'LotShape', 'LowQualFinSF', 'MasVnrArea', 'MasVnrType', 'MiscFeature',
                'MiscVal', 'MoSold', 'OpenPorchSF', 'PavedDrive', 'PoolArea', 'PoolQC', 'RoofMatl', 'RoofStyle', 'Street', 'TotRmsAbvGrd', 'TotalBsmtSF', 'Utilities', 'YrSold']

class AddEstimatates(TransformerMixin):
    """ Add a column from a prior estimator to the data """
    def __init__(self, estimator, col_name):
        self.estimator = estimator
        self.col_name = col_name

    def fit(self, X, y=None):
        """ Fit the estimator """
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        """ Add the estimator's prediction to the data """
        X = X.copy()
        X[self.col_name] = self.estimator.predict(X)
        return X

class TreeModelTransformer(TransformerMixin):
    """ Prepare the data for a tree model """

    def fit (self, X, y=None):  # pylint: disable=unused-argument
        """ Nothing to fit """
        return self

    def transform(self, df):
        """ Transform the data in prep for a tree model """

        # Drop categorical columns after one-hot encoding or labele encoding a select few
        # Adjust the skewness of a few numeric columns
        # After this is run, we'll look at feature importance and drop the ones that don't help

        # Label encoding...
        for col_name in ["ExterQual", "KitchenQual", "GarageQual", "BsmtQual", "GarageCond", "HeatingQC", "FireplaceQu"]:
            df[col_name] = df[col_name].map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'NA':0}, na_action='ignore')
            assert df[col_name].isnull().sum() == 0, f"{col_name} still has {int(df[col_name].isnull().sum()/df.shape[0] * 100)}% nulls)"
            assert np.issubdtype(df[col_name], np.number), f"{col_name} is not a number"
        df["Electrical"] = df["Electrical"].map({'SBrkr':4, 'FuseA':3, 'FuseF':2, 'FuseP':1, 'Mix':0})

        # Onehot encoding
        df["CentralAir"] = df.CentralAir.apply(lambda x: 1 if x == 'Y' else 0)              # Central air adds $500 (not much, but it helps)

        # Drop columns that have have been shown to not help in order to improves linear model performance
        df.drop(self.drop_cols, axis=1, inplace=True)

        return df

    @property
    def drop_cols(self):
        """ Return a list of columns that get dropped because they help neither linear model """
        return ['Alley', 'BedroomAbvGr', 'BldgType', 'BsmtFinSF2', 'BsmtFinType2', 'BsmtUnfSF', 'Condition1', 'Condition2', 'Electrical', 'EnclosedPorch',
                'Exterior2nd', 'ExterCond', 'ExterQual', 'Exterior1st',
                'FullBath', 'Functional', 'GarageCond', 'Heating', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotShape', 'LowQualFinSF', 
                'MasVnrArea', 'MoSold', 'MiscFeature', 'MiscVal', 'PoolQC', 'RoofMatl', 'PavedDrive', 'RoofStyle', 'Street', 'Utilities', 'WoodDeckSF', 'YrSold'
        ]

def get_onehot_encoder ():
    """ Return a onehot encoder that only applies to categorical columns, and leaves numeric columns alone """
    return Pipeline([
        ('column_splitter', FeatureUnion([
            ('numeric', Pipeline(steps=[
                    ('select_numeric', TypeSelector(np.number)),
            ])),
            ('categorical', Pipeline(steps=[
                    ('select_categorical', TypeSelector("object")),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])),
        ]))
    ])

def evaluate_models(df, label):
    """ Iteratively evaluate the two linear models as changes are made """
    ridge = Pipeline(steps=[
        ('data_cleaner', DataCleaner()),
        ('linear_model_transformer', LinearModelTransformer()),
        ('one_hot_encoder', get_onehot_encoder()),
        ('scaler', RobustScaler()),
        ('ridge', Ridge(alpha=9.7))
        ])

    lasso = Pipeline(steps=[
        ('data_cleaner', DataCleaner()),
        ('linear_model_transformer', LinearModelTransformer()),
        ('one_hot_encoder', get_onehot_encoder()),
        ('scaler', RobustScaler()),
        ('lasso', Lasso(alpha=0.0005))
        ])

    gb = Pipeline(steps=[
        ('add_linear_estimates', AddEstimatates(ridge, "linear_model_estimates")),
        ('data_cleaner', DataCleaner()),
        ('tree_model_transformer', TreeModelTransformer()),
        ('one_hot_encoder', get_onehot_encoder()),
        ('scaler', RobustScaler()),
        ('GB', GradientBoostingRegressor()) # n_estimators=1000, learning_rate=.01, max_depth=3, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber'))
        ])
    
    histgb = Pipeline(steps=[
        ('add_linear_estimates', AddEstimatates(lasso, "linear_model_estimates")),
        ('data_cleaner', DataCleaner()),
        ('tree_model_transformer', TreeModelTransformer()),
        ('one_hot_encoder', get_onehot_encoder()),
        ('scaler', RobustScaler()),
        ('GB', HistGradientBoostingRegressor()) # n_estimators=1000, learning_rate=.01, max_depth=3, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber'))
        ])

    model_name, model = "Ridge", ridge
    evaluate_model(model_name, model, df, label)
    #print_feature_importances(model, df, label, dropped_cols=ridge.named_steps["linear_model_transformer"].drop_cols)

    model_name, model = "Lasso", lasso
    evaluate_model(model_name, model, df, label)
    #print_feature_importances(model, df, label, dropped_cols=lasso.named_steps["linear_model_transformer"].drop_cols)

    """
    # Next, let's see if we can get a tree based model to fine tune the residuals of the linear model
    model_name, model = "Gb", gb
    gb.fit(df.drop(label, axis=1), df[label])
    evaluate_model(model_name, model, df, label)
    print_feature_importances(model, df, label, dropped_cols=gb.named_steps["tree_model_transformer"].drop_cols)

    model_name, model = "HistGb", histgb
    gb.fit(df.drop(label, axis=1), df[label])
    evaluate_model(model_name, model, df, label)
    print_feature_importances(model, df, label, dropped_cols=gb.named_steps["tree_model_transformer"].drop_cols)
    """

def evaluate_model(model_name, model, df, label, cv=5):
    """ Evaluate the model using cross validation """ 
    scores = cross_val_score(model, df.drop(label, axis=1), df[label], cv=cv, scoring='neg_mean_squared_error')
    print(f"{model_name} CV RMSE: {np.sqrt(np.mean(-scores)):.4f}")

def evaluate_model(model_name, model, df, label, cv=5):
    """ Alternative version, since cross_val_score doesn't give good error messages when I have a bug in the pipeline """
    from sklearn.model_selection import KFold
    kf = KFold(cv)
    mses = []
    for train_index, test_index in kf.split(df):
        X_train, X_test = df.drop(label, axis=1).iloc[train_index], df.drop(label, axis=1).iloc[test_index]
        y_train, y_test = df[label].iloc[train_index], df[label].iloc[test_index]
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        mses.append(mean_squared_error(yhat, y_test))
    print(f"{model_name} CV RMSE: {np.mean(mses)**.5:.4f}")

def do_it():
    """ Run the experiment """
    df = pd.read_csv("kaggle_housing/data_train.csv")
    label = "SalePrice"
    df[label] = np.log(df[label])       # Log transform the target variable, per competition rules; real goal should be mean absolute percentage error...
    evaluate_models(df, label)

if __name__ == "__main__":
    do_it()
