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
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler # StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split

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
        # Update: None of these seem to be used in our final model (they all get dropped), so we don't need to worry about them
        df["Functional"].fillna('Typ', inplace=True)
        df["Utilities"].fillna('None', inplace=True)
        df["MSZoning"].fillna('Unknown', inplace=True)
        df["SaleType"].fillna('Unknown', inplace=True)
        df["Exterior1st"].fillna('Unknown', inplace=True)
        df["Exterior2nd"].fillna('None', inplace=True)

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

class CategoricTransformer(BaseEstimator, TransformerMixin):
    """ Transform categorical features to numeric features """
    def __init__(self, is_linear):
        self.ran = False
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
        df["HasTenisCounrt"] = df.MiscFeature.apply(lambda x: 1 if x == 'TenC' else 0)
        df.drop("MiscFeature", axis=1, inplace=True)        # 96% nulls, and only TenisCounrt is useful. TODO - consider dropping HasTenisCounrt as well

        #['LotShape', 'HouseStyle', 'MasVnrType', 'Foundation']
        # CentralAir, GarageFinish
        df["PartialSalesCondition"] = df.SaleCondition.apply(lambda x: 1 if x == 'Partial' else 0)  # Partial sales are more expensive
        df["HasCentralAir"] = df.CentralAir.apply(lambda x: 1 if x == 'Y' else 0)                  # Central air adds $500 (not much, but it helps)
        df["ZoneCommercial"] = df["MSZoning"].apply (lambda x: 1 if x == 'C (all)' else 0)
        df["StreetPaved"] = df["Street"].apply (lambda x: 1 if x == 'Pave' else 0)

        # Drop some categorical columns for now - we'll add them back in later as they prove useful
        drop_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'LotConfig', 'LandSlope', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'Foundation',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
       'Electrical', 'Functional', 'GarageType',
       'GarageFinish', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition', "Utilities"]
        for col in drop_cols:
            if isinstance(df[col].dtype, np.number):
                print (col)
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
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
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
        ('categorical_transformer', CategoricTransformer(is_linear)),
        ('column_transformer', column_transformer),
        ('model', model)
    ])

    return pipeline

def evaluate_models():
    """ Create various models and evaluate them """
    train_csv = "kaggle_housing/data_train.csv"
    label = "SalePrice"

    df = pd.read_csv(train_csv)

    # Scale the target column (SalePrice) by taking the log, to remove skewness and not penalize predictions on less expensive houses
    df.SalePrice = np.log(df.SalePrice)

    # Split the data into training and test sets
    X_train, X_val, y_train, y_val = train_test_split(df.drop(label, axis=1), df[label], test_size=0.2, random_state=12)

    models = [('Lasso with alpha=.005', Lasso(alpha=.005), True),
            ('Ridge', Ridge(), True),
            ('GB', GradientBoostingRegressor(), False),
            ('RF', RandomForestRegressor(n_estimators=300, n_jobs=-1), False)
    ]

    pipelines = {}

    for name, model, is_linear in models:
        print (f"\n{name}")
        pipeline = create_pipeline(model, is_linear)
        pipeline.fit(X_train.copy(), y_train)
        pipelines[name] = pipeline

        yhat_train = pipeline.predict(X_train.copy())
        yhat_val = pipeline.predict(X_val.copy())
        print_metrics(yhat_train, y_train, yhat_val, y_val)

    print ("\nEnsemble: all models")
    ensemble_prediction = np.mean([pipeline.predict(X_val.copy()) for pipeline in pipelines.values()], axis=0)
    print_metrics(ensemble_prediction, y_val, ensemble_prediction, y_val)

    print ("\nEnsemble: Ridge + GB")
    ensemble_prediction = np.mean([pipelines[model].predict(X_val.copy()) for model in ["Ridge", "GB"]], axis=0)
    print_metrics(ensemble_prediction, y_val, ensemble_prediction, y_val)

    # print the most important features from the final (random forest) model
    #print ("\nTraining feature importance:")
    #print_feature_importances(X_train, y_train, pipeline)
    #print ("\nValidation feature importance:")
    #print_feature_importances(X_val, y_val, pipeline)

    # Predict on the test data
    #test_csv = "kaggle_housing/data_test.csv"
    #df_test = pd.read_csv(test_csv)
    #pipeline.predict(df_test)

if __name__ == "__main__":
    evaluate_models()
