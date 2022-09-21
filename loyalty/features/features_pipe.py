"""Create pipeline for loading, merging and cleaning dataset"""

import logging
from features.features import (get_features, FeatureUnionReframer)
from utils import dump_pipeline, load_pipeline
from kb import ParamsValues

logger = logging.getLogger(__name__)
pipeline_filenames = ParamsValues.pipeline_filenames


class FeaturesProcessor():
    """Processor module."""
    def __init__(self,
                 model_name=None,
                 periodic_features=None,
                 skip_init=False):
        """Init method.

        Args:
            model_name: str
        """
        self.model_name = model_name
        self.periodic_features = periodic_features
        if not skip_init:
            self.__set_features_pipelines()

    def __set_features_pipelines(self):
        """Set features pipelines."""
        self.periodic_features_pipe = FeatureUnionReframer \
            .make_df_retaining(get_features(self.periodic_features))

    def __load_features_pipelines(self):
        """load features pipelines."""

        self.periodic_features_pipe = load_pipeline(pipeline_filenames['periodic'])

    def process(self, ground_truth, df_periodic):
        """Preprocess features and targets.

        Args:
            ground_truth: pd.DataFrame
            df_periodic: pd.DataFrame

        Return:
            output: pd.DataFrame, a dataframe containing features & target.
        """
        def process_pipeline(pipeline, train, val, name):
            """ Generic pipeline runner"""
            # Fit and transform train data. Transform only validation
            train = pipeline.fit_transform(train)
            val = pipeline.transform(val)
            dump_pipeline(pipeline, name)
            return train, val

        # Filter train and validation accounts
        train_accts = ground_truth[ground_truth.dataset == 0]
        val_accts = ground_truth[ground_truth.dataset == 1]

        # Split period data
        df_periodic_train = df_periodic[df_periodic.acct.isin(train_accts.acct)]
        df_periodic_val = df_periodic[df_periodic.acct.isin(val_accts.acct)]

        logger.info('Building periodic features.')
        df_periodic_train, df_periodic_val = process_pipeline(
            pipeline=self.periodic_features_pipe,
            train=df_periodic_train,
            val=df_periodic_val,
            name=pipeline_filenames['periodic'],
        )

        logger.info('Merging features and target datasets.')
        output_train = train_accts.merge(df_periodic_train, on=['acct', 'period'], how='inner')
        output_val = val_accts.merge(df_periodic_val, on=['acct', 'period'], how='inner')

        output_train = output_train.drop(['dataset'], axis=1)
        output_val = output_val.drop(['dataset'], axis=1)
        logger.info(f'Shape train {output_train.shape}')
        logger.info(f'Shape val {output_val.shape}')

        return output_train, output_val

    def process_predict(self, df_periodic):
        """Preprocess features and targets.

        Args:
            df_periodic: pd.DataFrame

        Return:
            output: pd.DataFrame, a dataframe containing features.
        """

        logger.info('Building features.')
        df_periodic = self.periodic_features_pipe.transform(df_periodic)
        
        logger.info('Merging features and target datasets.')
        # Merge non periodic features
        output = df_periodic
        logger.info(f'Shape {output.shape}')
        # Features that need both datasets

        logger.info(f'Shape {output.shape}')
    
        n_months = ParamsValues.minimum_required_periods
        logger.info(f'Drop the first {n_months} periods per account.')
        output = output.drop(output.groupby('acct').head(n_months).index)

        return output

    @classmethod
    def import_pipelines(cls):
        """Import pipelines from piclke and return a loaded feature processor class."""

        feat_pipe = cls(skip_init=True)

        logger.info('Loading pipelines.')
        feat_pipe.__load_features_pipelines()

        return feat_pipe
