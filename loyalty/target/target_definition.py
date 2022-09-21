"""Process raw data to define target."""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from preprocessing.preprocess_transformers import PositionBalanceTransformer
from kb import ParamsValues

logger = logging.getLogger(__name__)


class TargetProcessor(BaseEstimator, TransformerMixin):
    """Raw data processing to define target"""
    def __init__(self, filename='position_balance_sum.csv', closing_treshold=100):
        """Args:
            filename: str
            closing_treshold: int, indicates the minimum value that the TMV can reach before an
            account is considered as closed.
        """
        self.filename = filename
        self.closing_treshold = closing_treshold

    def fit(self, X):
        """Fit method."""
        return self
    @staticmethod
    def account_aggregation(X):
        """
            This function take an positional balance dataframe and compute some functions to aggregate
            required values into an account DataFrame.
            Args:
             Input: Positional balance's dataframe
             Output: Account's data frame
        """
        # Take last period
        last_period = X.period.unique().max()

        # Add auxiliar columns per account
        df_acct = X.groupby('acct').agg(
            q_periods = pd.NamedAgg(column='period',
                                    aggfunc='size'),
            q_closing_periods = pd.NamedAgg(column='closing_cond', 
                                            aggfunc='sum'),
            first_closing_cond = pd.NamedAgg(column='closing_cond', 
                                             aggfunc='first'),
            last_closing_cond = pd.NamedAgg(column='closing_cond', 
                                            aggfunc='last'),
            max_period = pd.NamedAgg(column='period', 
                                            aggfunc='max')

        ).reset_index()

        # Filter accounts with at least one not closed period
        df_acct = df_acct[df_acct['q_periods']!=df_acct['q_closing_periods']]

        # Set acct as index
        df_acct = df_acct.set_index('acct')

        # Defining when an account cloused because its tmv
        df_acct['acct_closed_tmv'] = ((df_acct['first_closing_cond'] == 0) & 
                              (df_acct['last_closing_cond'] == 1)).astype(int)
        # Not valid accounts, porque no es válida
        df_acct['not_valid_acct'] = ((df_acct['first_closing_cond'] == 1) & 
                                     (df_acct['last_closing_cond'] == 0)).astype(int)
        # Closed accounts because not apearing in following periods
        df_acct['acct_closed_per'] = (df_acct['max_period']!=last_period).astype(int)
        
    
        return df_acct[['acct_closed_tmv','not_valid_acct','acct_closed_per','max_period']]

    def transform(self, X, y=None):
        """Define target per account and period.
        Args:
            X: pd.DataFrame with acct, period, mkt, cash and pos columns. Period must have
            format 'yyyymm'.
        Returns:
            pd.DataFrame with columns: acct, period, target
        """
        if not all([c in X.columns for c in ['acct', 'period', 'mkt', 'cash', 'pos']]):
            raise ValueError("Missing column in DataFrame")

        X = X[['acct', 'period', 'mkt', 'cash', 'pos']]

        logger.info('Filtering invalid rows')

        # Ensure period is string type
        X.period = X.period.astype(str)

        # Sort dataframe apropriately
        X = X.sort_values(['acct', 'period'])
        
        # positional balance processing 
        logger.info('START CLEANING, using PositionalBalanceTransformer')
        pbt = PositionBalanceTransformer()
        X = pbt.transform(X)
        
        # Create closing condition, values under this threshold are marked
        closing_cond = X['mkt'] < self.closing_treshold
        X['closing_cond'] = closing_cond.astype(int)
        
        # Get all account aggregation
        df_acct = self.account_aggregation(X)

        
        # DF_CLEAN
        # Filter valid accounts
        df_clean = X[X.acct.isin(df_acct.index.unique())]
        
        # Merge data
        df_clean = df_clean.merge(df_acct, how='left', on=['acct'])
        
        # Cummulated closing condition
        closing_cond_cumsum = X.groupby(['acct', 'period'])['closing_cond']\
            .sum().to_frame('closing_cond_cumsum').groupby(level=0)\
            .cumsum().reset_index()
        df_clean = df_clean.merge(closing_cond_cumsum, how='left', on=['acct', 'period'])

        df_clean['closing_cond_cumsum_aux_t-1'] = df_clean\
        .groupby('acct')['closing_cond_cumsum'].shift(-1)

        
        # TARGET DEFINITION    
        #Features necesaries from acct [acct_closed_tmv, not_valid_account, acct_closed_per]

        # Period churn
        df_clean['churn_acct_per'] = (df_clean['acct_closed_per']
                                      & (df_clean['period'] == df_clean['max_period'])).astype(int)

        # TMV Churn
        df_clean['churn_acct_tmv'] = ((df_clean['closing_cond_cumsum_aux_t-1'] == 1)
                                      & (df_clean['acct_closed_tmv'] == 1)).astype(int)
        
        # Create target column
        logger.info(f'Creating target column')
        churned_accounts_tmv = df_clean.groupby('acct')['churn_acct_tmv']\
            .sum().loc[lambda x: x > 0].index
        df_clean['target'] = np.where(df_clean.acct.isin(churned_accounts_tmv),
                                      df_clean['churn_acct_tmv'],
                                      df_clean['churn_acct_per']).astype(int)

        ground_truth = df_clean[(df_clean['not_valid_acct'] == 0)
                                & (df_clean['closing_cond_cumsum'] == 0)]

        # Keep needed columns
        ground_truth = ground_truth[['acct', 'period', 'target']]

        # Drop last period
        last_period = ground_truth.period.unique().max()
        logger.info(f'Dropping last period {last_period}')
        ground_truth = ground_truth[~ground_truth.period.isin([last_period])]

        return ground_truth


class TargetSelector(BaseEstimator, TransformerMixin):
    """Target selector per account"""
    def __init__(self, picking_method='random', selected_period=None):
        """Args:
            filename: str
        """
        self.picking_method = picking_method
        self.selected_period = selected_period

        if picking_method == 'one_period' and not selected_period:
            raise ValueError(f'Selected period must be not None. Got {selected_period}')

    def fit(self, X):
        """Fit method."""
        return self

    def transform(self, X, y=None):
        """Select one period per account using different techniques and target column
        Args:
            X: pd.DataFrame with acct, period, target columns. Period must have
            format 'yyyymm'.
        Returns:
            pd.DataFrame with columns: acct, period, target
        """
        logger.info('Selecting target.')
        if self.picking_method == 'random':
            logger.info('Picking target: random')

            positive_class = X[X['target'] == 1].groupby('acct').head(1)
            ch_acc = X[X['target'] == 1]['acct'].unique()
            negative_class = X[~X['acct'].isin(ch_acc)].groupby('acct').sample(1, random_state=42)

            # Concat positive and negative classes
            X = pd.concat([positive_class, negative_class])

            # Shuffle rows
            X = X.sample(frac=1, random_state=42).reset_index(drop=True)
        elif self.picking_method == 'last':
            logger.info('Picking target: last')

            X = X.sort_values('period', ascending=False).groupby('acct').head(1)
        elif self.picking_method == 'one_period':
            logger.info(f'Picking target: one period. Period: {self.selected_period}')

            X = X[X['period'] == self.selected_period]
        else:
            raise ValueError(f'Not implemented picking method: {self.picking_method}')

        return X


class TargetSplitter(BaseEstimator, TransformerMixin):
    """Target train/test split"""
    def __init__(self, test_size=0.10, random_state=42):
        """Args:
            filename: str
        """
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X):
        """Fit method."""
        return self

    def transform(self, X, y=None):
        """Select one period per account using different techniques and target column
        Args:
            X: pd.DataFrame with acct, period, target columns. Period must have
            format 'yyyymm'.
        Returns:
            pd.DataFrame with columns: acct, period, target
        """
        X_all = X[['acct', 'period']]
        y_all = X['target']

        logger.info('Spliting train/val dataset')
        X_train, X_val, y_train, y_val = train_test_split(X_all,
                                                          y_all,
                                                          test_size=0.10,
                                                          stratify=y,
                                                          random_state=42)

        output = X[['acct', 'period', 'target']].copy()

        # Create column with dataset information
        output['dataset'] = 0
        output.loc[output.acct.isin(X_val.acct), 'dataset'] = 1

        return output
