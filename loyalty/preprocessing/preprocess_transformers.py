"""Transformers for loading, merging and cleaning dataset"""
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from kb import ParamsValues
from utils import load_data

logger = logging.getLogger(__name__)


class PositionBalanceTransformer(BaseEstimator, TransformerMixin):
    """Position balance transformer."""
    def __init__(self, filename='position_balance_sum.csv', is_predict=None):
        """
        Args:
            feature: str
        """
        self.columns_names = ['median_tmv', 'max_tmv']
        self.filename = filename
        self.is_predict = is_predict

    def fit(self, X, y=None):
        """Fit method.

        Args:
            X: pd.DataFrame.

        Returns:
            self.
        """
        return self

    @staticmethod
    def filter_periods(X, periods_key):
        """Filter non wanted periods

        Args:
            X: pd.DataFrame with column period.

        Returns:
            Filtered pd.DataFrame.
        """

        if "period" not in X.columns:
            raise ValueError("Period column not in DataFrame")

        PERIODS = ParamsValues.features_static_values.get(periods_key)

        logger.info('Filtering periods')

        before = X.shape[0]
        X = X[X.period.isin(PERIODS)]
        after = X.shape[0]

        logger.info(f'Before total rows: {before:,d}')
        logger.info(f'After total rows: {after:,d}')
        logger.info(f'Dropped {before-after:,d} rows')

        return X

    @staticmethod
    def filter_negative_accounts(X, epsilon=-1):
        """Filter negative accounts

        Args:
            X: pd.DataFrame with columns: acct, period, cash, mkt, pos.
            epsilon: float, indicates the limit since we consider a balance as negative

        Returns:
            Filtered pd.DataFrame.
        """
        if not all([c in X.columns for c in ['acct', 'period', 'cash', 'mkt', 'pos']]):
            raise ValueError("Missing column in DataFrame")

        logger.info('Filtering negative accounts')

        before = X.shape[0]
        before_accounts = X['acct'].nunique()

        # Get negative accounts
        filters = (X.mkt < epsilon) | (X.cash < epsilon) | (X.pos < epsilon)
        neg_accounts = X[filters]['acct'].unique()

        # Filter
        X = X[~X.acct.isin(neg_accounts)]

        after = X.shape[0]
        after_accounts = X['acct'].nunique()

        logger.info(f'Before total accounts: {before_accounts:,d}')
        logger.info(f'Dropped {(1-after_accounts/before_accounts):.2%} of accounts')
        logger.info(f'After total accounts: {after_accounts:,d}')
        logger.info(f'Dropped {before-after:,d} rows')

        return X

    @staticmethod
    def filter_new_accounts(X):
        """Filter new accounts

        Args:
            X: pd.DataFrame with columns: mkt, pos, cash, period, account
            n_months: int, minimum required quantity of periods per account

        Returns:
            Filtered pd.DataFrame.
        """

        if not all([c in X.columns for c in ['acct', 'period']]):
            raise ValueError("Missing column in DataFrame")

        n_months = ParamsValues.minimum_required_periods

        logger.info(f'Keep accounts with more than {n_months} months')

        before = X.shape[0]
        before_accounts = X['acct'].nunique()

        # Create new column with number of periods per account
        # TODO: review this.
        # A value is trying to be set on a copy of a slice from a DataFrame
        X = X.copy()

        X['q_periods'] = X[['acct', 'period']].groupby('acct').transform('count')

        # Keep accounts with more than {n_months} months
        X = X[X.q_periods > n_months]

        X.drop(columns=['q_periods'], inplace=True)

        after = X.shape[0]
        after_accounts = X['acct'].nunique()

        logger.info(f'Before total accounts: {before_accounts:,d}')
        logger.info(f'Dropped {(1-after_accounts/before_accounts):.2%} of accounts')
        logger.info(f'After total accounts: {after_accounts:,d}')
        logger.info(f'Dropped {before-after:,d} rows')

        return X

    @staticmethod
    def filter_check_mkt_pos_cash(X, tolerance=100):
        """Filter inconsistent accounts

        Args:
            X: pd.DataFrame with columns: acct, period, cash, mkt, pos.
            tolerance: maximum tolerated difference between mkt and [cash + pos]

        Returns:
            Filtered pd.DataFrame.
        """
        if not all([c in X.columns for c in ['acct', 'period', 'cash', 'mkt', 'pos']]):
            raise ValueError("Missing column in DataFrame")

        logger.info('Filtering inconsistent accounts')

        before = X.shape[0]
        before_accounts = X['acct'].nunique()

        # Get inconsistent accounts
        X['mkt-pos-cash'] = X['mkt'] - (X['pos'] + X['cash'])
        ck_consistency = (np.abs(X['mkt-pos-cash']) < tolerance)
        acct_inconsistent_tmv = X[~ck_consistency].acct.unique()

        # Filter
        X = X[~X.acct.isin(acct_inconsistent_tmv)]

        after = X.shape[0]
        after_accounts = X['acct'].nunique()

        logger.info(f'Before total accounts: {before_accounts:,d}')
        logger.info(f'Dropped {(1-after_accounts/before_accounts):.2%} of accounts')
        logger.info(f'After total accounts: {after_accounts:,d}')
        logger.info(f'Dropped {before-after:,d} rows')

        return X

    def transform(self, X):
        """Transform input to compute median and max tmv per account

        Args:
            X: pd.DataFrame with columns: acct, period, cash, mkt, pos

        Returns:
            Filtered and cleaned pd.DataFrame
        """

        if not all([c in X.columns for c in ['acct', 'period', 'cash', 'mkt', 'pos']]):
            raise ValueError("Missing column in DataFrame")

        logger.info('Processing position balance')

        # Ensure period is string type
        X.period = X.period.astype(str)

        if not self.is_predict:
            periods_key = 'valid_modeling_periods'
        else:
            periods_key = 'predict_periods'

        X = self.filter_negative_accounts(X)
        X = self.filter_periods(X, periods_key)
        X = self.filter_new_accounts(X)
        
        # Reset index after filtering the DataFrame
        X = X.reset_index(drop=True)

        # TODO: Remove this when max_mkt feature is removed
        features = X

        # Compute max features
        tmp = X[['acct', 'mkt']].groupby('acct').rolling(3,
                                                         min_periods=1).max().reset_index(drop=True)
        tmp.columns = ['feature_max_mkt']

        features = pd.concat([features, tmp], axis=1)

        features['feature_prev_1_max_mkt'] = features[[
            'acct', 'feature_max_mkt'
        ]].groupby('acct').feature_max_mkt.shift(1).reset_index(drop=True)
        features['feature_prev_2_max_mkt'] = features[[
            'acct', 'feature_max_mkt'
        ]].groupby('acct').feature_max_mkt.shift(1).reset_index(drop=True)

        return features

    def get_feature_names(self):
        return self.columns_names



class CashMovementTransformer(BaseEstimator, TransformerMixin):
    """ Cash movements processing."""
    def __init__(self, filename='cash.csv'):
        """Args:
            filename: str
        """
        self.filename = filename

    def fit(self, X):
        """Fit method."""
        return self

    def transform(self, X):
        """Compute cash movements by account

        Args:
            X: pd.DataFrame with account and period column.

        Returns:
            pd.DataFrame with extra columns: cash_mov, cash_mov_roll_3,
            feature_cash_sig_3
        """

        # Reading data
        logger.info('Loading cash information')
        try:
            df_cash = load_data(file_name=self.filename)
        except FileNotFoundError as fnf_error:
            logger.info(fnf_error)
            return X

        # Fill null values
        df_cash = df_cash.fillna(0)

        # Sum up columns and change the sign.
        # Positive sign means cash in, negative sign means cash out
        df_cash['cash_mov'] = (-1) * (df_cash.iloc[:, 2:].sum(axis=1))

        # Cast period type
        df_cash.period = df_cash.period.astype(str)
        
        df_cash.acct = df_cash.acct.astype(str)
        X.acct = X.acct.astype(str)
        # Merge files
        merged = X.merge(df_cash[['acct', 'period', 'cash_mov']], on=['acct', 'period'], how='left')
        merged = merged.fillna(0)

        # Rolling window using the last x months - MKT
        window_size = 3
        mkt_roll = merged.groupby('acct').shift(1).rolling(
            window=window_size).mean().reset_index().mkt
        merged['mkt_roll'] = mkt_roll

        # Rolling window using the last x months - CASH
        cash_mov_roll = merged[['acct', 'cash_mov'
                                ]].groupby(['acct'
                                            ]).rolling(window_size).sum().reset_index().cash_mov
        merged['cash_mov_roll_3'] = cash_mov_roll

        # Calculate cash significance
        merged['feature_cash_sig_3'] = round(merged.cash_mov_roll_3 / merged.mkt_roll, 3)

        # Fill null
        merged = merged[['acct', 'period', 'cash_mov', 'cash_mov_roll_3',
                         'feature_cash_sig_3']].fillna(0)
        logger.info('Merging cash movements information to position_balance_sum')

        return pd.merge(X, merged, on=['acct', 'period'], how='left')