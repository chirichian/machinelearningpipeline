"""Run build features pipelines"""

import click
import logging
import os
from features.features_pipe import FeaturesProcessor
from utils import data_path, load_datasets
from kb import ParamsValues


@click.command(name='build_features')
def build_features_cli():
    """ Build features from datasets.

        Returns:
            pd.dataframe with the datasets processed and merged.
        """
    logger.info('Loading datasets.')
    ground_truth, df_periodic = load_datasets()

    logger.info('Processing datasets.')
    processor = FeaturesProcessor(
        periodic_features=ParamsValues.processor_params.get('periodic_features'),
        )

    train_data, validation_data, = processor.process(
        ground_truth=ground_truth,
        df_periodic=df_periodic,
    )

    logger.info('Dumping preprocessed data.')
    processed_dir = data_path('processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    train_data.to_csv(os.path.join(processed_dir, 'features_dataset_train.csv'),
                      sep='|',
                      index=False)
    validation_data.to_csv(os.path.join(processed_dir, 'features_dataset_validation.csv'),
                           sep='|',
                           index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    build_features_cli()