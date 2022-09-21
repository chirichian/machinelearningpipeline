"""Predict pipeline client"""

import click
import logging
import os
from preprocessing.preprocess_pipe import Preprocessor
from features.features_pipe import FeaturesProcessor
from utils import data_path, load_account_data
from kb import ParamsValues
from models.model import ClassifierModel
import pandas as pd

filenames = ParamsValues.predict_filenames


@click.command(name='predict')
def predict_cli():
    """ Preprocess, build features and predict from scratch.

        Returns:
            pd.dataframe with the datasets processed and merged.
        """
    logger.info('Loading accounts data to predict.')
    df_account = load_account_data()

    logger.info('Preprocessing accounts data.')
    preprocessor = Preprocessor.import_pipelines()
    df_periodic = preprocessor.preprocess_predict(df_account)

    logger.info('Processing datasets.')
    processor = FeaturesProcessor.import_pipelines()
    processed_data = processor.process_predict(df_periodic=df_periodic)

    logger.info('Dumping features data.')
    processed_dir = data_path('processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    processed_data.to_csv(os.path.join(processed_dir, filenames['features']), sep='|', index=False)

    processed_data = pd.read_csv(os.path.join(processed_dir, filenames['features']), sep='|')
    cat_cols = processed_data.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        processed_data[c] = processed_data[c].astype('category')

    logger.info('Loading model.')
    model = ClassifierModel.import_model()
    y_predicted = model.predict(processed_data)

    logger.info('Dumping prediction result data.')
    prediction_result = processed_data[['acct', 'period']].copy()
    prediction_result['y_pred'] = y_predicted
    prediction_result.to_csv(os.path.join(processed_dir, filenames['predictions']),
                             sep='|',
                             index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    predict_cli()
