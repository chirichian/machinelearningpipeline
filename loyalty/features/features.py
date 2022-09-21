""" Feature transformers. """

import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder as OneHotEncoderSkLearn

logger = logging.getLogger(__name__)


class IdentityFeatures(BaseEstimator, TransformerMixin):
    """Take the data as feature."""
    def __init__(self, columns_names):
        """Args:
            columns_names: array like, containing columns to use.
        """
        self.columns_names = columns_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns_names]

    def get_feature_names(self):
        return self.columns_names


class FeatureEncoding(BaseEstimator, TransformerMixin):
    """Encode data with dict"""
    def __init__(self, column_name, dict_):
        """Args:
            column_name: array like, containing columns to use.
            dict_: dict. Values of the DataFrame to replace
        """
        self.column_name = column_name
        self.dict_ = dict_

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column_name].replace(self.dict_)

    def get_feature_names(self):
        return self.column_name


class AcctAgeFeature(BaseEstimator, TransformerMixin):
    """Given a an open period and a reference period with format 'yyyymm',
    obtain difference in periods to calculate the age of account in months."""
    def __init__(self, columns_names, output_name):
        """Args:
            columns_names: array like of lenght 2, containing columns
            to use. First one must be period of refference, second one
            the openning period.
            output_name: str, name of the output column.
        """
        self.columns_names = columns_names
        self.output_name = output_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.columns_names]
        open_period = pd.to_datetime(X.iloc[:, 1], format='%Y%m')
        open_period_yr = open_period.dt.year
        open_period_mth = open_period.dt.month
        ref_period = pd.to_datetime(X.iloc[:, 0], format='%Y%m')
        ref_period_yr = ref_period.dt.year
        ref_period_mth = ref_period.dt.month
        acct_age = (ref_period_yr - open_period_yr) * 12 + (ref_period_mth - open_period_mth)
        return acct_age.to_frame(name=self.output_name)

    def get_feature_names(self):
        return [self.output_name]


class TopCategoriesFeature(BaseEstimator, TransformerMixin):
    """Given a category column, it returns top n categories or categories above threshold"""
    def __init__(self, column_name, threshold=None, topn=None, category_name=None):
        """Args:
            column_name: name of the column.
            threshold: int. minimun value of observations by category. If not none,
                it keeps categories above this value.
            topn: int. number of categories. If not none, it keeps at the most the top n
                categories.
            category_name: name to group small categories.
        """
        self.column_name = column_name
        self.threshold = threshold
        self.topn = topn
        self.category_name = category_name
        self.top_categories = None

    def fit(self, X, y=None):
        X = X[[self.column_name]]

        if self.threshold:
            self.top_categories = X[self.column_name].value_counts(
            ).loc[lambda x: x > self.threshold].index.tolist()[:self.topn]
        else:
            self.top_categories = X[self.column_name].value_counts().index.tolist()[:self.topn]

        return self

    def transform(self, X, y=None):
        X = X[[self.column_name]]

        X.loc[~X[self.column_name].isin(self.top_categories), self.column_name] = self.category_name

        return X

    def get_feature_names(self):
        return [self.column_name]


class MeanFeature(BaseEstimator, TransformerMixin):
    """Compute the mean value of a column using a period of n months."""
    def __init__(self, columns_names, n_months, id_col='acct', pipeline=False, output_names=None):
        """Args:
            columns_names: array like, containing columns to use.
            n_months: int. Number of months to use as a window period
            id_col: str. Name of the column used to groupby the dataframe.
                Default value `acct`
            pipeline: bool. If True the id_col is not dropped.
                This is useful when working with pipelines
            output_names: array like containing output column names.
                If None default value is `mean_` + column name
        """
        self.columns_names = columns_names
        self.id_col = id_col
        self.n_months = n_months
        self.pipeline = pipeline

        if output_names:
            self.output_names = output_names
        else:
            self.output_names = ['mean_' + c for c in columns_names]

        # If True, add id_col to output cols
        if self.pipeline:
            self.output_names.append(self.id_col)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Make empty DataFrame
        df = X[[self.id_col]].copy().reset_index(drop=True)

        for out_c, c in zip(self.output_names, self.columns_names):
            tmp = X[[self.id_col] + [c]].groupby(self.id_col).rolling(
                self.n_months, min_periods=1).mean().reset_index(drop=True)
            tmp.columns = [out_c]

            df = pd.concat([df, tmp], axis=1)

        return df[self.output_names]

    def get_feature_names(self):
        return self.output_names


class ShiftFeature(BaseEstimator, TransformerMixin):
    """Shift the value of a column to get the previous month
    """
    def __init__(self,
                 columns_names,
                 n_months,
                 id_col='acct',
                 output_names=None,
                 keep_original_cols=False):
        """Args:
            columns_names: array like, containing columns to use.
            n_months: int. Number of months to shift
            id_col: str. Name of the column used to groupby the dataframe.
                Default value `acct`
            output_names: array like containing output column names.
                If None default value is `shift_m{n_months}_` + column name
            keep_original_cols: bool. If true columns_names are added to the
                output columns
        """
        self.columns_names = columns_names
        self.n_months = n_months
        self.id_col = id_col
        self.keep_original_cols = keep_original_cols

        if output_names:
            self.output_names = output_names
        else:
            self.output_names = [f'shift_m{self.n_months}_' + c for c in columns_names]

        if self.keep_original_cols:
            # Add columns_names to output cols
            self.output_names += self.columns_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Compute feature

        Args:
            X: pd.DataFrame with `column_names`

        Returns:
            pd.DataFrame with output_names columns
        """
        cols = [self.id_col] + self.columns_names if self.keep_original_cols else [self.id_col]

        # Make empty DataFrame
        df = X[cols].copy().reset_index(drop=True)

        for out_c, c in zip(self.output_names, self.columns_names):
            tmp = X[[self.id_col] + [c]].groupby(self.id_col).shift(
                self.n_months).reset_index(drop=True)
            tmp.columns = [out_c]

            df = pd.concat([df, tmp], axis=1)

        return df[self.output_names]

    def get_feature_names(self):
        return self.output_names


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Given parameters, encode each parameter by their values."""
    def __init__(self, column_names):
        """"Args:
            column_names: array-like, columns to one hot encode.
        """
        self.column_names = column_names
        self.encoder = OneHotEncoderSkLearn(handle_unknown='ignore')

    def fit(self, X, y=None):
        X = X[self.column_names]
        self.encoder.fit(X, y)
        return self

    def transform(self, X, y=None):
        X = X[self.column_names]
        return self.encoder.transform(X)

    def get_feature_names(self):
        names = self.encoder.get_feature_names(input_features=self.column_names)
        return names


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Given parameters and target value, encode each parameter value
     by using the mean of the target"""
    def __init__(self, column_names, target):
        """Args:
            column_names: array-like, columns to encode
            target: string, target to use as encoder.
        """
        self.column_names = column_names
        self.target = target
        self.dicts_ = {}

    def fit(self, X, y=None):
        for col in self.column_names:
            dict_ = X.groupby(col)[self.target].mean().to_dict()
            self.dicts_[col] = dict_
        return self

    def transform(self, X, y=None):
        for col in self.column_names:
            X[f'te_{col}'] = X[col].map(lambda val: self.dicts_[col].get(val, None))
        return X[[f'te_{col}' for col in self.column_names]]

    def get_feature_names(self):
        return [f'te_{col}' for col in self.column_names]


class TrendTransformer(BaseEstimator, TransformerMixin):
    """Calculate linear regression of a parameter anb use the slope as trend."""
    def __init__(self, column_names, months):
        """Args:
            column_names: array-like, columns to calculate the trend.
            months: array-like, months to look-back and calculate trend.
        """
        self.column_names = column_names
        self.months = months

    def fit(self, X, y=None):
        return self

    def local_trend(self, n, values):
        """Calculate linear regression and return slope
        Args:
            n: int, amount of periods to calculate the linear regression
            values: array-like, values to regress.
        Return:
            slope, the slope of the linear regression.
            """
        coefficients = np.polyfit(range(n), values, deg=1)
        return coefficients[0]

    def transform(self, X, y=None):
        self.output_names = [f'trend_{col}_{i}' for col in self.column_names for i in self.months]
        for col in self.column_names:
            for i in self.months:
                str_col = f'trend_{col}_{i}'
                X[str_col] = self.local_trend(i, [
                    X.fillna(0).groupby('acct')[col].shift(i - 1 - j).fillna(method='bfill')
                    for j in range(i)
                ])
        return X[self.output_names]

    def get_feature_names(self):
        return self.output_names


class SumLastMonthsFeature(BaseEstimator, TransformerMixin):
    """Take a column and calculate the sum whithin n past months."""
    def __init__(self, columns_names, shift_qty):
        """Args:
            columns_names: Name of the columns
            shift_qty: qty of past months to sum
        """
        self.columns_names = columns_names
        self.shift_qty = shift_qty
        self.output_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.columns_names]

        # Verify acct and period are in X
        if any(n not in X.columns for n in ['acct', 'period']):
            raise KeyError("Unable to create feature due to missing column period or acct")

        # calculate sum
        ref_col = X.columns.difference(['acct', 'period']).tolist()
        for col in ref_col:
            col_name = col + '_sum_m' + str(self.shift_qty)
            self.output_names.append(col_name)
            X[col_name] = X.groupby(['acct']).rolling(self.shift_qty,
                                                      min_periods=1).sum()[col].tolist()

        return X[self.output_names]

    def get_feature_names(self):
        return self.output_names


class MaxLastMonthsFeature(BaseEstimator, TransformerMixin):
    """Take a column and calculate the sum whithin n past months."""
    def __init__(self, columns_names, shift_qty):
        """Args:
            columns_names: Name of the columns
            shift_qty: qty of past months to evaluate
        """
        self.columns_names = columns_names
        self.shift_qty = shift_qty
        self.output_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.columns_names]

        # Verify acct and period are in X
        if any(n not in X.columns for n in ['acct', 'period']):
            raise KeyError("Unable to create feature due to missing column period or acct")

        # calculate max value
        ref_col = X.columns.difference(['acct', 'period']).tolist()
        for col in ref_col:
            col_name = col + '_max_m' + str(self.shift_qty)
            self.output_names.append(col_name)
            X[col_name] = X.groupby(['acct']).rolling(self.shift_qty,
                                                      min_periods=1).max()[[col]].reset_index()[col]
        return X[self.output_names]

    def get_feature_names(self):
        return self.output_names


class VariationsFeature(BaseEstimator, TransformerMixin):
    """Take a column and calculate variation whithin n past months."""
    def __init__(self, columns_names, shift_qty):
        """Args:
            columns_names: Name of the columns
            shift_qty: qty of past months to shift
        """
        self.columns_names = columns_names
        self.shift_qty = shift_qty
        self.output_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.columns_names]

        # Verify acct and period are in X
        if any(n not in X.columns for n in ['acct', 'period']):
            raise KeyError("Unable to create feature due to missing column period or acct")

        # Calculate variations
        ref_col = X.columns.difference(['acct', 'period']).tolist()
        for col in ref_col:
            # shift the column
            X['shifted'] = X.groupby(['acct']).shift(self.shift_qty)[col]

            # calculate the ratio
            col_name = col + '_ratio_m' + str(self.shift_qty)
            self.output_names.append(col_name)
            X[col_name] = X['shifted'] / X[col]

        return X[self.output_names]

    def get_feature_names(self):
        return self.output_names


class FeatureUnionReframer(TransformerMixin, BaseEstimator):
    """Transforms preceding FeatureUnion's output back into Dataframe."""
    def __init__(self, feat_union, cutoff_transformer_name=True):
        """Args:
            feat_union: sklearn.FeatureUnion to applies the columns
            cutoff_transformer_name: cut transformer name
        """
        self.union = feat_union
        self.cutoff_transformer_name = cutoff_transformer_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("FeatureUnion's output to DataFrame")
        assert isinstance(X, np.ndarray)
        if self.cutoff_transformer_name:
            cols = ['__'.join(c.split('__')[1:]) for c in self.union.get_feature_names()]
        else:
            cols = self.union.get_feature_names()
        df = pd.DataFrame(data=X, columns=cols)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    @classmethod
    def make_df_retaining(cls, feature_union):
        """With this method a feature union will be returned as a pipeline
        where the first step is the union and the second is a transformer that
        re-applies the columns to the union's output."""
        return Pipeline([('union', feature_union), ('reframe', cls(feature_union))])


class PipelineFeature(BaseEstimator, TransformerMixin):
    """Take a list of transformers to compute one feature"""
    def __init__(self, feature_transformers):
        """Args:
            features: List of features
        """
        self.feature_transformers = make_pipeline(*feature_transformers)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Compute
        """
        return self.feature_transformers.transform(X)

    def get_feature_names(self):
        return self.feature_transformers[-1].get_feature_names()


features_by_names = {
    'periodic_key':
    lambda: IdentityFeatures(['acct', 'period']),
    'cash':
    lambda: IdentityFeatures(['cash']),
    'mkt':
    lambda: IdentityFeatures(['mkt']),
    'feature_mean_mkt':
    lambda: MeanFeature(columns_names=['mkt'], n_months=3),
    'feature_prev_1_mean_mkt':
    lambda: PipelineFeature([
        MeanFeature(columns_names=['mkt'], n_months=3, pipeline=True),
        ShiftFeature(columns_names=['mean_mkt'], n_months=1),
    ], ),
    'feature_prev_2_mean_mkt':
    lambda: PipelineFeature([
        MeanFeature(columns_names=['mkt'], n_months=3, pipeline=True),
        ShiftFeature(columns_names=['mean_mkt'], n_months=2),
    ], ),
    'feature_max_mkt':
    lambda: IdentityFeatures(['feature_max_mkt']),
    'feature_prev_1_max_mkt':
    lambda: IdentityFeatures(['feature_prev_1_max_mkt']),
    'feature_prev_2_max_mkt':
    lambda: IdentityFeatures(['feature_prev_2_max_mkt']),
    'cash_mov':
    lambda: IdentityFeatures(['cash_mov']),
    'cash_mov_roll_3':
    lambda: IdentityFeatures(['cash_mov_roll_3']),
    'feature_cash_sig_3':
    lambda: IdentityFeatures(['feature_cash_sig_3']),
    'qty_in':
    lambda: IdentityFeatures(['qty_in']),
    'qty_out':
    lambda: IdentityFeatures(['qty_out']),
    'qty_in_mean':
    lambda: IdentityFeatures(['qty_in_mean']),
    'qty_out_mean':
    lambda: IdentityFeatures(['qty_out_mean']),
    'feature_qty_out_dif':
    lambda: IdentityFeatures(['feature_qty_out_dif']),
    'feature_qty_in_dif':
    lambda: IdentityFeatures(['feature_qty_in_dif']),
    'feature_twrr_non_adjusted':
    lambda: IdentityFeatures(['feature_twrr_non_adjusted']),
    'sd_twrr':
    lambda: IdentityFeatures(['sd_twrr']),
    'medians':
    lambda: IdentityFeatures(['medians']),
    # Sum last periods
    'sum_m2':
    lambda: SumLastMonthsFeature([
        'acct',
        'period',
        'cash_mov',
        'qty_in',
        'qty_out',
    ], shift_qty=2),
    'sum_m3':
    lambda: SumLastMonthsFeature([
        'acct',
        'period',
        'cash_mov',
        'qty_in',
        'qty_out',
    ], shift_qty=3),
    # Ratios
    'ratio_m1':
    lambda: VariationsFeature([
        'acct',
        'period',
        'cash',
        'mkt',
        'sd_twrr',
        'feature_twrr_non_adjusted',
    ],
                              shift_qty=1),
    'ratio_m2':
    lambda: VariationsFeature([
        'acct',
        'period',
        'cash',
        'mkt',
        'sd_twrr',
        'feature_twrr_non_adjusted',
    ],
                              shift_qty=2),
    'ratio_m3':
    lambda: VariationsFeature([
        'acct',
        'period',
        'cash',
        'mkt',
        'sd_twrr',
        'feature_twrr_non_adjusted',
    ],
                              shift_qty=3),
    # Max last periods
    'max_m3':
    lambda: MaxLastMonthsFeature([
        'acct',
        'period',
        'qty_in',
        'qty_out',
    ], shift_qty=3),
    'max_m2':
    lambda: MaxLastMonthsFeature([
        'acct',
        'period',
        'qty_in',
        'qty_out',
    ], shift_qty=2),
    'cash_trends':
    lambda: TrendTransformer(['cash'], [2, 3]),
    'mkt_trends':
    lambda: TrendTransformer(['mkt'], [2, 3]),
    'twrr_trends':
    lambda: TrendTransformer(['sd_twrr'], [2, 3]),
    'medians_trends':
    lambda: TrendTransformer(['medians'], [2, 3]),
    'twrr_non_adjusted_trends':
    lambda: TrendTransformer(['feature_twrr_non_adjusted'], [2, 3]),
    # Non periodic features
    'non_periodic_key':
    lambda: IdentityFeatures(['acct']),
    'open_period':
    lambda: IdentityFeatures(['open_period']),
    'acct_type':
    lambda: IdentityFeatures(['acct_type']),
    # Objective
    'objective':
    lambda: FeatureEncoding(['objective'], dict(income=0, balanced=1, growth=2)),
    'income':
    lambda: IdentityFeatures(['income']),
    'balanced':
    lambda: IdentityFeatures(['balanced']),
    'growth':
    lambda: IdentityFeatures(['growth']),
    # Risk
    'max_risk':
    lambda: FeatureEncoding(['max_risk'], dict(low=0, medium=1, high=2)),
    'low':
    lambda: IdentityFeatures(['low']),
    'medium':
    lambda: IdentityFeatures(['medium']),
    'high':
    lambda: IdentityFeatures(['high']),
    # Client  information
    'est_age':
    lambda: IdentityFeatures(['est_age']),
    'liquid_assets':
    lambda: IdentityFeatures(['liquid_assets']),
    'fixed_assets':
    lambda: IdentityFeatures(['fixed_assets']),
    # Target encoding
    'target_enc_id':
    lambda: IdentityFeatures([
        'prov',
        'country_code',
        'sex',
        'occu',
        'est_worth',
        'est_earn',
        'in_know',
    ]),
    'city':
    lambda: TopCategoriesFeature('city', threshold=50),
    'pcode':
    lambda: TopCategoriesFeature('pcode', threshold=40),
    # features after merge
    'account_age':
    lambda: AcctAgeFeature(['period', 'open_period'], 'account_age'),
    'target_enc':
    lambda: TargetEncoder([
        'prov',
        'country_code',
        'sex',
        'occu',
        'in_know',
    ], 'target'),
}


def get_features(features):
    """Get features by name.

    Args:
        features: array like, containing features to process.

    Return:
        FeatureUnion: sklearn feature union to process features.
    """

    if not features:
        names = features_by_names.keys()
    else:
        names = [f for f in features if isinstance(f, str)]
    # validate that names exists
    if any(n not in features_by_names for n in names):
        raise KeyError("Valid features are: {}".format(', '.join(sorted(features_by_names.keys()))))
    # if no features were given, all features by name are included

    named_features = [(name, features_by_names[name]()) for name in names]
    # make a big union
    return FeatureUnion(transformer_list=named_features)
