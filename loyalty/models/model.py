"""Train and test predictive churn model"""

from os import makedirs
from os.path import exists

import datetime
import joblib
import lightgbm as ltb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from models.log_experiment import log_experiment
from kb import ParamsValues
from utils import data_path

import logging
logger = logging.getLogger(__name__)


class ClassifierModel():
    """
    This class is in charge of train,
    validate and save the statistical model
    """
    def __init__(self, ltb_params=None, save_logg=True, training_version=None):
        """ Run the complete experiment.
        Args:
            ltb_params: dict. Values to be used in model
            save_logg: bool. Flag to save output files to disk
            training_version: Str. Model version name
        """
        self.final_test_metrics = None
        self.model = None
        self.save_logg = save_logg
        self.params = {
            'ltb_params': {},
            'threshold': ParamsValues.training_params['model']['threshold'],
            'nfolds': ParamsValues.training_params['model']['nfolds'],
        }
        # TODO: Add output paths parameters in init class
        self.outputs = {
            'ltb_model': ParamsValues.training_params['output']['ltb_model'],
            'xtrain': ParamsValues.training_params['output']['xtrain'],
            'ytrain': ParamsValues.training_params['output']['ytrain'],
            'xval': ParamsValues.training_params['output']['xval'],
            'yval': ParamsValues.training_params['output']['yval'],
        }

        if ltb_params:
            self.params['ltb_params'].update(ltb_params)

        if training_version:
            self.training_version = training_version
        else:
            self.training_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Split X in train/val data, Train model, and save results

        Args:
            X: pd.DataFrame with features
            y: pd.DataFrame with target
        Returns:
            None
        """
        # TODO: Check X.shape[0] == y.shape[0]

        # Correlation Analysis
        self.corr_analysis(X=X_train, version=self.training_version)

        # TODO: Implement a Cross-Validation
        
        # Drop acct and period
        X_train = X_train.drop(columns=['acct', 'period'])
        X_val = X_val.drop(columns=['acct', 'period'])

        # Model Training
        self.model = ltb.LGBMClassifier(**self.params['ltb_params'])
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )

        if self.save_logg:
            results = self.model.evals_result_
            # TODO: this only works with one metric
            metric = ParamsValues.training_params['model']['ltb_params']['metric'][0]
            results = pd.DataFrame({
                'train_auc': results['training'][metric],
                'test_auc': results['valid_1'][metric],
            })

            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            y_pred = self.predict(X_val)
            self.compute_metrics(y_true=y_val, y_pred=y_pred)
            
            # Plot AUC ROC and AUC PR:
            self.plot_auc_metrics(y_val, y_pred, self.training_version)

            # Set max_rows to None to display all rows in file
            pd.set_option('display.max_rows', None)

            # Log the experiment results:
            log_experiment(
                description=f"Prediction report of learning model.\n"
                f"Holdout Performance: .\n"
                f"auc={self.final_test_auc} - prauc={self.final_test_prauc}",
                tag="modeling_results",
                version=self.training_version,
                number=self.training_version,
                # Log all the results:
                records={
                    "Dataset Train Shape": X_train.shape,
                    "Dataset Validation Shape": X_val.shape,
                    "Holdout Auc": self.final_test_auc,
                    "Holdout PrAuc": self.final_test_prauc,
                    "Parameters": self.params,
                    "Support Metrics": self.final_test_metrics,
                    "Confusion Matrix": pd.DataFrame(self.conf_matrix),
                    "Classification Report": self.report,
                    "Features": X_train.columns,
                    "Feature Importance": feature_importance,
                    "Training Results": results.tail(),
                })

        # Save model
        self.save_model()

        # Save datasets
        self.save_dataset(X_train, artifact_path=self.outputs['xtrain'])
        self.save_dataset(y_train, artifact_path=self.outputs['ytrain'])
        self.save_dataset(X_val, artifact_path=self.outputs['xval'])
        self.save_dataset(y_val, artifact_path=self.outputs['yval'])

    def compute_metrics(self, y_true, y_pred):
        """
        Compute metrics

        Args:
            y_true:
            y_pred:
        Returns:
            None
        """

        self.final_test_auc = roc_auc_score(y_true, y_pred)
        self.final_test_prauc = average_precision_score(y_true, y_pred)
        self.conf_matrix = confusion_matrix(y_true, y_pred > self.params['threshold'])
        self.report = classification_report(y_true, y_pred > self.params['threshold'])

        # Save metrics
        self.final_test_metrics = {
            'accuracy': accuracy_score(y_true, y_pred > self.params['threshold']),
            'precision': precision_score(y_true, y_pred > self.params['threshold']),
            'recall': recall_score(y_true, y_pred > self.params['threshold']),
            'f1': f1_score(y_true, y_pred > self.params['threshold'])
        }

    def predict(self, X_test, y_test=None):
        """
        Compute predictions for X_test dataset

        Args:
            X_test: pd.DataFrame
            y_test: (optional). pd.DataFrame
        Returns:
            array-like with predicted values
        """
        if not self.model:
            # Error model not trained
            # TODO: Raise exception
            pass

        # Drop acct and period columns, if exists
        X_test = X_test.drop(columns=["acct", "period"], errors='ignore')

        # y_true = y_test
        y_pred = self.model.predict(X_test)

        return y_pred

    def predict_proba(self, X_test, y_test=None):
        """
        Compute predictions for X_test dataset

        Args:
            X_test: pd.DataFrame
            y_test: (optional). pd.DataFrame
        Returns:
            array-like of shape = [n_samples, n_classes] with predicted values
        """
        if not self.model:
            # Error model not trained
            # TODO: Raise exception
            pass

        # Drop acct and period columns, if exists
        X_test = X_test.drop(columns=["acct", "period"], errors='ignore')

        # y_true = y_test
        y_pred = self.model.predict_proba(X_test)

        return y_pred

    def cross_validation(self, X_train, y_train):
        """
        Perform a 10-fold cross validation with XGBoost
        """
        dtrain = ltb.Dataset(X_train, label=y_train)
        history = ltb.cv(
            self.params['ltb_params'],
            dtrain,
            nfold=self.params['nfolds'],
            seed=42,
            stratified=True,
            verbose_eval=False,
        )

        history = pd.DataFrame({'auc-mean': history['auc-mean'], 'auc-stdv': history['auc-stdv']})

        if self.save_logg:
            log_experiment(description="Cross validation report.\n",
                           tag="cross_validation",
                           version=self.training_version,
                           number=self.training_version,
                           records={
                               "Cross validation": history,
                           })

    def corr_analysis(self, X, version):
        """"
        This function obtains the correlation matrix between features
        and stores it in the experiment log, additionally saves a
        figure with a heatmap of the matrix and also returns
        a pandas DataFrame where are the top correlation per feature.
        """
        # Remove target column
        np_df = X.iloc[:, 1:]
        # Creating Heatmap of correlations and saving it
        corr = np_df.corr()
        corr_plot = sns.heatmap(corr)

        # The destination folder must exist
        if not exists(data_path('logs', version)):
            makedirs(data_path('logs', version))

        output_path = data_path('logs', version, 'plot.corr.{}.png'.format(version))

        corr_plot.figure.savefig(output_path, bbox_inches="tight")

        # Creating the top correlations Dataframe
        corr = corr.stack().reset_index()
        corr.columns = ['c1', 'c2', 'value']
        # Removing the diagonal
        corr = corr[corr.c1 != corr.c2].sort_values(by='value', ascending=False)
        output_path = data_path('logs', version, 'df.corr.{}.csv'.format(version))
        corr.to_csv(output_path, sep='|')

    def plot_auc_metrics(self, y_true, y_pred, version):
        """
        Plot the ROC AUC and the PR AUC on the holdout set
        """
        roc_auc = roc_auc_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        roc_display = RocCurveDisplay(fpr=fpr,
                                      tpr=tpr,
                                      roc_auc=roc_auc,
                                      estimator_name="Account Close").plot()

        prec, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
        pr_display = PrecisionRecallDisplay(precision=prec,
                                            recall=recall,
                                            average_precision=pr_auc,
                                            estimator_name="Account Close").plot()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)

        output_path = data_path('logs', version, 'plot.auc.{}.png'.format(version))
        pr_display.figure_.savefig(output_path)

    def save_model(self, artifact_path=None):
        """
        Save model object to a file.
        """

        if not artifact_path:
            artifact_path = self.outputs['ltb_model']
        joblib.dump(value=self.model, filename=artifact_path)

    def __load_model(self, artifact_path=None):
        """
        Load model object to class variable
        """

        if not artifact_path:
            artifact_path = self.outputs['ltb_model']
        self.model = joblib.load(filename=artifact_path)

    @classmethod
    def import_model(cls):
        """
        Import model and return ModelTraining class
        """

        model_training = cls()
        model_training.__load_model()

        return model_training

    def save_dataset(self, dataset, artifact_path):
        """
        Save given dataset in specified location

        Args:
            dataset (dataframe): object to save
            artifact_path (str): where to save it
        """
        logger.info("About to upload to {}".format(artifact_path))
        try:
            dataset.to_csv(artifact_path, sep='|')
            logger.info('Dataset saved {}'.format(artifact_path))
        except (IOError, KeyError, PermissionError, FileNotFoundError) as e:
            logger.info('Error trying to save dataset {}'.format(e))
