"""Run preprocess pipelines"""

import logging
import os
import click

from utils import data_path, load_account_data
from preprocessing.preprocess_pipe import Preprocessor


@click.command(name='make_dataset')
def make_dataset_cli():
    """Make datasets from raw context files.

    Returns:
        pd.dataframe with the datasets processed and merged.
    """
    logger.info('Loading accounts data.')
    df_account = load_account_data()

    logger.info('Preprocessing accounts data.')
    preprocessor = Preprocessor()
    ground_truth, preprocessed_data = preprocessor.preprocess(df_account)

    processed_dir = data_path('processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    logger.info(f'Dumping preprocessed data to: {processed_dir}')
    logger.info(f'Ground truth shape: {ground_truth.shape}')
    ground_truth.to_csv(os.path.join(processed_dir, 'ground_truth.csv'), sep='|', index=False)

    logger.info(f'Periodic dataset shape: {preprocessed_data.shape}')
    preprocessed_data.to_csv(os.path.join(processed_dir, 'periodic_data.csv'), sep='|', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    make_dataset_cli()
