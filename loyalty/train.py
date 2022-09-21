"""Script to run churn model training"""

import ast
from os.path import exists
from os import makedirs

import click
import logging
import pandas as pd

from models import model
from kb import ParamsValues

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(name='train')
@click.option('--ltb_params',
              default='{}',
              callback=lambda ctx, param, value: ast.literal_eval(value),
              help=('ltb params. Must be a dict in a string format. ' +
                    'Example: --ltb_params "{\'num_leaves\': 100}"'))
@click.option('--save_logg', type=bool, default=True)
def model_training_script(save_logg, ltb_params):
    """
    Generic Script that will train and log the model.
    """

    # The destination folders must exist
    folders = ParamsValues.training_params['output']['folders']
    for f in folders:
        if not exists(f):
            makedirs(f)

    # Feature Pre-processing and selection:
    path = ParamsValues.training_params['input_train']
    logger.info(f'Reading data from {path}')

    # Read training data
    X_train = pd.read_csv(path, sep='|')

    # Convert all str columns to category type
    cat_cols_train = X_train.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols_train:
        X_train[c] = X_train[c].astype('category')
    logger.info('Shape of dataset {}'.format(X_train.shape))

    # Split features and target
    y_train = X_train['target']
    X_train = X_train.drop(columns='target')

    path = ParamsValues.training_params['input_val']
    logger.info(f'Reading data from {path}')

    # Read training data
    X_val = pd.read_csv(path, sep='|')

    # Convert all str columns to category type
    cat_cols_val = X_val.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols_val:
        X_val[c] = X_val[c].astype('category')

    y_val = X_val['target']
    X_val = X_val.drop(columns='target')

    ltb_default_params = ParamsValues.training_params['model']['ltb_params']
    if ltb_params:
        ltb_default_params.update(ltb_params)

    logger.info(f"Executing the model...")
    model_training = model.ClassifierModel(ltb_params=ltb_default_params, save_logg=save_logg)
    model_training.fit(X_train, y_train, X_val, y_val)


if __name__ == '__main__':
    model_training_script()
