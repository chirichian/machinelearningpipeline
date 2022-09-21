"""Create pipeline for loading, merging and cleaning dataset"""

import logging
from sklearn.pipeline import make_pipeline

from preprocessing.preprocess_transformers import (
    PositionBalanceTransformer,
    CashMovementTransformer
    )
from target.target_definition import (
    TargetProcessor,
    TargetSelector,
    TargetSplitter
)

from kb import ParamsValues
from utils import dump_pipeline, load_pipeline

logger = logging.getLogger(__name__)
pipeline_filenames = ParamsValues.pipeline_filenames


class Preprocessor():
    """Preprocessor module."""
    def __init__(self, model_name=None, skip_init=None):
        """Init method.

        Args:
            model_name: str
        """
        self.model_name = model_name
        if not skip_init:
            self.__set_preprocess_pipeline()

    def __set_preprocess_pipeline(self):
        """Set preprocess pipeline."""
        posbal_transformer = PositionBalanceTransformer()
        cashmov_transformer = CashMovementTransformer()
        target_processor = TargetProcessor()
        target_selector = TargetSelector(
            picking_method=ParamsValues.target_params['picking_method'],
            selected_period=ParamsValues.target_params['selected_period'],
        )
        target_splitter = TargetSplitter()

        self.preprocess_pipe_groundtruth = make_pipeline(
            target_processor,
            target_selector,
            target_splitter,
        )
        # Preprocess periodic data
        self.preprocess_pipe = make_pipeline(
            posbal_transformer,
            cashmov_transformer,
        )
        

    def __load_pipelines(self):
        """Load preprocess pipeline."""
        self.preprocess_pipe = load_pipeline(pipeline_filenames['md_periodic'])

    def preprocess(self, data):
        """Preprocess features and targets.

        Args:
            data: pandas dataframe.
        """
        preprocessed_periodic_data = self.preprocess_pipe.fit_transform(data)
        dump_pipeline(self.preprocess_pipe, pipeline_filenames['md_periodic'])

        preprocessed_ground_truth = self.preprocess_pipe_groundtruth.fit_transform(data)

        return preprocessed_ground_truth, preprocessed_periodic_data

    def preprocess_predict(self, data):
        """Preprocess features.

        Args:
            data: pandas dataframe.
        """
        self.preprocess_pipe.set_params(positionbalancetransformer__is_predict=True)
        preprocessed_periodic_data = self.preprocess_pipe.transform(data)

        return preprocessed_periodic_data

    @classmethod
    def import_pipelines(cls):
        """Import pipelines from piclke and return a loaded feature processor class."""

        prepoc_pipe = cls(skip_init=True)

        logger.info('Loading pipelines.')
        prepoc_pipe.__load_pipelines()

        return prepoc_pipe
