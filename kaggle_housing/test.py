"""
Experiment with linear regression models for the housing data
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
from sklearn.preprocessing import OneHotEncoder, RobustScaler # StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score

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

def print_feature_importances(X, y, model):
    """ Calculate feature importance by checking the RMSE for each column when it is set to it's mean value """
    importance = {}
    X_copy = X.copy()
    yhat = model.predict(X_copy)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    importance["All columns"] = rmse    # Baseline; useful columns will have higher rmse (since the mean isn't good enough)
    for col in X.columns:
        X_copy = X.copy()
        # For numeric columns, set the value to the mean of the column
        if np.issubdtype(X_copy[col].dtype, np.number):
            X_copy[col] = X_copy[col].mean()
        # For categorical columns, set the value to the most frequent value in the column
        else:
            X_copy[col] = X_copy[col].mode()[0]
        yhat = model.predict(X_copy)
        rmse = np.sqrt(mean_squared_error(y, yhat))
        importance[col] = rmse

    for col, rmse in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print (f"{col}: {rmse:.4f}")

class AveragePredictor(BaseEstimator):
    """ Average the prediction of serveral models """
    def __init__(self, models):
        self.models = models

    def fit(self, X, y=None):
        """ Fit all  models """
        for model in self.models:
            model.fit(X.copy(), y)
        return self

    def predict(self, X):
        """ Average the predictions of all models """
        yhat = np.mean([model.predict(X.copy()) for model in self.models], axis=0)
        return yhat

class TypeSelector(BaseEstimator, TransformerMixin):
    """ Pipeline componenet to select columns of a particular type; needed for column transformers """
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        """ Fit the selector; nothing to do"""
        return self

    def transform(self, X):
        """ Transform the data by selecting only columns of the specified type """
        return X.select_dtypes(include=[self.dtype])

class DataCleaner(BaseEstimator, TransformerMixin):
    """ Clean up the input data based on docs to remove NaNs, adjust data types, etc """
    def fit(self, X, y=None):
        """ Nothing to fit """
        return self

    def transform(self, df):
        """ Preprocess the data """

        # Drop useless Id column (it is not a predictor); if alread dropped, then transformation has already run once and doesn't need to run again
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
        df["Functional"].fillna('Typ', inplace=True)        # Not currently used in final model
        df["Utilities"].fillna('None', inplace=True)        # Dropped in final model
        df["MSZoning"].fillna('Unknown', inplace=True)      # Only used in final model if = C (all); deduct for ZoneCommercial
        df["SaleType"].fillna('Unknown', inplace=True)      # Not currently used in final model
        df["Exterior1st"].fillna('Unknown', inplace=True)   # Not currently used in final model
        df["Exterior2nd"].fillna('None', inplace=True)      # Not currently used in final model

        null_cols = set(col for col in df.columns if df[col].isnull().sum() > 0)
        for col in null_cols:
            print (f"Warning: column {col} has {df[col].isnull().sum()/df.shape[0]:.2%} nulls")

        return df

class LabelEncoder(BaseEstimator, TransformerMixin):
    """ Some categorical features can be label-encoded (mapped to a single int value) """
    def __init__(self):
        self.ran = False
        self.dropped_columns = []

    def fit(self, X, y=None):
        """ Fit the imputer; nothing to do"""
        return self

    def transform(self, df):
        """ Convert 'quality' and 'condition' categorical features into numeric values from 4 (excellent) to 0 (poor) """
        for col_name in ["ExterQual", "KitchenQual", "GarageQual", "BsmtQual", "GarageCond", "HeatingQC", "FireplaceQu"]:
            df[col_name] = df[col_name].map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'NA':0}, na_action='ignore')
            assert df[col_name].isnull().sum() == 0, f"{col_name} still has {int(df[col_name].isnull().sum()/df.shape[0] * 100)}% nulls)"
            assert np.issubdtype(df[col_name], np.number), f"{col_name} is not a number"

        df["Electrical"] = df["Electrical"].map({'SBrkr':4, 'FuseA':3, 'FuseF':2, 'FuseP':1, 'Mix':0})

        return df

class UnskewFeatures (BaseEstimator, TransformerMixin):
    """ Normalize skewed numeric features """
    def __init__(self):
        self.ran = False

    def fit(self, X, y=None):
        """ Fit the imputer; nothing to do"""
        return self

    def transform(self, df):
        """ Normalize skewed numeric data """
        for cols in ["GrLivArea", "LotArea", "1stFlrSF"]:
            df[cols] = np.log(df[cols])

        return df
    
class DropOrTransform(BaseEstimator, TransformerMixin):
    """ Certain columns should be either dropped entirely or transformed to something else """
    def __init__(self, is_linear):
        self.is_linear = is_linear  # Behaviour is different for linear vs tree-based models

    def fit(self, X, y=None):
        """ Fit the imputer; nothing to do"""
        return self

    def transform(self, df):
        """ Preprocess the data """

        # Total baths is a much better predictor than individual bath counts
        df["TotalBathsAg"] = df["FullBath"] + .5* df["HalfBath"]
        df["TotalBathBsmt"] = df["BsmtFullBath"] + .5 * df["BsmtHalfBath"]
        df.drop(["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], axis=1, inplace=True)

        # Hand-craft a few features from the categorical data

        # OneHotEncode a few categorical values before dropping the feature
        if self.is_linear:
            df["SalesCondition"] = df.SaleCondition.apply(lambda x: 1 if x == 'Partial' else 0 if x == 'Normal' else -1)
            df["KitchenAbvGr"] = df["KitchenAbvGr"].apply (lambda x: 1 if x == 1 else 0)

        df["EstateSale"] = df.SaleType.apply(lambda x: 1 if x == 'COD' else 0)              # Estate sales are cheaper
        df["IsNew"] = df.SaleType.apply(lambda x: 1 if x == 'New' else 0)                   # New buildings are more expensive
        df["CentralAir"] = df.CentralAir.apply(lambda x: 1 if x == 'Y' else 0)              # Central air adds $500 (not much, but it helps)
        df["ZoneCommercial"] = df["MSZoning"].apply (lambda x: 1 if x == 'C (all)' else 0)

        # Drop some categorical columns
        drop_cols = ["Alley", "BldgType", "BsmtFinType2", "Electrical", "Fence", "Heating",
                     "PoolQC", "RoofMatl", "RoofStyle", "Street", "Utilities"]   # Definite no's from exploratory analysis

        drop_cols += ["BsmtExposure", "BsmtFinType1", "Condition1", "Condition2", "Exterior1st", "Exterior2nd",
                      "Foundation", "Functional", "GarageFinish", "GarageType",
                    "BldgType", "LandContour", "LandSlope", "LotConfig", "LotShape",
                    "MSSubClass", "MSZoning", "MasVnrType", "PavedDrive", "SaleType"] # TBD if we'll use these or not

        # Also drop useless numeric columns
        drop_cols += ['3SsnPorch', 'PoolArea', 'MoSold', 'YrSold', 'BsmtFinSF2', 'EnclosedPorch', 'LowQualFinSF', 'LotFrontage', 'MiscVal']
        df.drop(drop_cols, axis=1, inplace=True)

        return df

def create_pipeline(model, is_linear):
    """ Pipeline for numerical features """
    if is_linear:
        num_pipeline = Pipeline(steps=[
            ('select_numeric', TypeSelector(np.number)),
            ('unskew_features', UnskewFeatures()),          # Unskew a few skewed numeric features
            ('scaler', RobustScaler())                      # Also try RobustScaler, StandardScaler, MinMaxScaler; XXX - skip for forest models...
        ])
    else:   # Don't need a scaler for tree-based models
        num_pipeline = Pipeline(steps=[
            ('select_numeric', TypeSelector(np.number)),
            ('unskew_features', UnskewFeatures()),
        ])

    # Pipeline for categorical features
    cat_pipeline = Pipeline(steps=[
        ('select_categorical', TypeSelector('object')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a column transformer to apply the pipelines to the appropriate columns
    column_transformer = Pipeline([
        ('column_transformer', FeatureUnion([
            ('categorical', cat_pipeline),
            ('numeric', num_pipeline)
        ]))
    ])

    # Final pipeline
    pipeline = Pipeline(steps=[
        ('data_cleaner', DataCleaner()),
        ('label_encoder', LabelEncoder()),
        ('drop_or_transform', DropOrTransform(is_linear)),
        ('column_transformer', column_transformer),
        ('model', model)
    ])

    return pipeline

def evaluate_model(model, df, label, model_name):
    """ Evaluate the model using cross validation """
    cv_score = cross_val_score(model, df.drop(label, axis=1), df[label], scoring='neg_mean_squared_error')
    print(f"{model_name} RMSE: {(-cv_score.mean())**.5:.4f}")

def evaluate_models():
    """ Create various models and evaluate them """
    train_csv = "kaggle_housing/data_train.csv"
    label = "SalePrice"

    df = pd.read_csv(train_csv)

    # Scale the target column (SalePrice) by taking the log, to remove skewness and not penalize predictions on less expensive houses
    df.SalePrice = np.log(df.SalePrice)

    models = [
        ('Lasso ', Lasso(alpha=.001), True),
        ('Ridge ', Ridge(alpha=9), True),
        ('GB    ', GradientBoostingRegressor(), False),
        ('HistGB', HistGradientBoostingRegressor(), False),
    ]

    pipelines = {}

    for name, model, is_linear in models:
        model = create_pipeline(model, is_linear)
        evaluate_model(model, df, label, name)
        pipelines[name] = model

    ensemble = AveragePredictor(list(pipelines.values()))
    evaluate_model(ensemble, df, label, "Ensemble")

    # print the most important features from the final (random forest) model
    print ("\nFeature importance:")
    ensemble.fit(df.drop(label, axis=1), df[label])
    #print_feature_importances(df.drop(label, axis=1), df[label], ensemble)

    # Predict on the test data
    test_csv = "kaggle_housing/data_test.csv"
    df_test = pd.read_csv(test_csv)
    ensemble.predict(df_test)

def tune_hyperparams():
    """ Create various models and evaluate them """
    train_csv = "kaggle_housing/data_train.csv"
    label = "SalePrice"

    df = pd.read_csv(train_csv)

    # Scale the target column (SalePrice) by taking the log, to remove skewness and not penalize predictions on less expensive houses
    df.SalePrice = np.log(df.SalePrice)

    # Split the data into training and test sets
    X_train, X_val, y_train, y_val = train_test_split(df.drop(label, axis=1), df[label], test_size=0.2, random_state=12)

    alpha_to_rmse = []
    best_alpha = None
    best_rmse = 1000000
    for alpha in np.linspace(7, 10, 50):
        model = create_pipeline(Ridge(alpha=alpha), True)
        model.fit(X_train.copy(), y_train)
        yhat_val = model.predict(X_val.copy())
        rmse = np.sqrt(mean_squared_error(yhat_val, y_val))
        alpha_to_rmse.append((alpha, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    print ("best alpha:", best_alpha)

    # Graph alpha_to_rmse
    import matplotlib.pyplot as plt
    alphas = [x[0] for x in alpha_to_rmse]
    rmses = [x[1] for x in alpha_to_rmse]
    plt.plot(alphas, rmses)
    plt.show()


    #model = create_pipeline(Lasso(), True)
    #if False:
    #    for key, value in model.get_params().items():
    #        print (key)
    #        print(value)
    #        print()

    #params = {'model__alpha': [0.001]}

    #from sklearn.model_selection import GridSearchCV
    #grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    #grid_search.fit(X_train, y_train)
    #print (grid_search.best_params_)
    #print (grid_search.best_score_)



if __name__ == "__main__":
    evaluate_models()
