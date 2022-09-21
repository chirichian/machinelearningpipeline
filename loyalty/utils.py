"""Utility function cross-project."""

import os
import pandas as pd
import joblib

DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),'data/'))


def data_path(*joins):
    """
    Data path getter

    Args:
        joins: Extra path to be join with the data path

    Returns:
        os.path data path
    """
    print(os.path.join(DATA_PATH, *joins))
    return os.path.join(DATA_PATH, *joins)


def load_data(file_name='mfclna.csv'):
    """
    Client file loader

    Args:
        file_name: (optional). Str. Filename to load

    Returns:
        pd.DataFrame with client data
    """
    # TODO: Change docstring and default values
    csv_path = data_path(file_name)
    df = pd.read_csv(csv_path)
    return df


def load_account_data(file_name='position_balance_sum.csv'):
    """
    Account file loader

    Args:
        file_name: (optional). Str. Filename to load

    Returns:
        pd.DataFrame with account data
    """

    csv_path = data_path(file_name)
    df = pd.read_csv(csv_path)
    # Ensure DataFrame is sorted
    df = df.sort_values(['acct', 'period'])

    return df


def load_datasets(ground_truth='ground_truth.csv',
                  periodic_dataset='periodic_data.csv',
                  ):
    """
    Account file loader

    Args:
        ground_truth: (optional). Str. Filename to load
        periodic_dataset: (optional). Str. Filename to load

    Returns:
        pd.DataFrame with account data
    """

    ground_truth = pd.read_csv(data_path('processed', ground_truth), sep='|')
    periodic_dataset = pd.read_csv(data_path('processed', periodic_dataset), sep='|')

    # Ensure period is string type
    ground_truth.period = ground_truth.period.astype(str)
    periodic_dataset.period = periodic_dataset.period.astype(str)

    # Ensure DataFrames are sorted
    ground_truth = ground_truth.sort_values(['acct', 'period'])
    periodic_dataset = periodic_dataset.sort_values(['acct', 'period'])

    return ground_truth, periodic_dataset, 


def dump_pipeline(pipeline, filename):
    """Dump pipeline to file.

    Args:
        pipeline: pipeline, pipeline to dump.
        filename: str, filename to use.
    """
    file_dir = data_path('processed', f'{filename}.joblib')
    joblib.dump(value=pipeline, filename=file_dir)


def load_pipeline(filename):
    """Load pipeline to file.

    Args:
        filename: str, pipeline filename to use.
    """
    file_dir = data_path('processed', f'{filename}.joblib')
    return joblib.load(filename=file_dir)
