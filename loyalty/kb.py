"""
Knowledge Base
--------------
This file containst all knowledge base of the project. This includes:
- Constants
- Static tables (E.g: to store Table names or codes)
"""
import numpy as np

from utils import data_path


class ParamsValues(object):

    minimum_required_periods = 1

    features_static_values = {
        'valid_modeling_periods': [
            '2021-01', 
            '2021-02', 
            '2021-03',
            '2021-04', 
            '2021-05',
            '2021-06', 
            '2021-07', 
            '2021-08', 
            '2021-09', 
            '2021-10'
        ],
        'predict_periods': [
            '2021-05',
            '2021-06', 
            '2021-07', 
            '2021-08', 
            '2021-09', 
            '2021-10'
        ]
    }

    target_params = {
        # 'random', 'last', 'one_period'
        'picking_method': 'random',
        'selected_period': None,
    }

    pipeline_filenames = {
        'periodic': 'periodic_pipe',
        'non_periodic': 'non_periodic_pipe',
        'after_merge': 'features_after_merge_pipe',
        'md_periodic': 'makedataset_periodic_pipe',
        'md_non_periodic': 'makedataset_non_periodic_pipe',
    }

    predict_filenames = {
        'features': 'features_predict.csv',
        'predictions': 'prediction_result.csv',
    }
    # There are just some features from broadridge repo
    processor_params = {
        'periodic_features': [
            'periodic_key',
            'cash',
            'mkt',
            #'feature_mean_mkt',
            #'feature_prev_1_mean_mkt',
            #'feature_prev_2_mean_mkt',
            'feature_max_mkt',
            'feature_prev_1_max_mkt',
            'feature_prev_2_max_mkt',
            #'cash_mov',
            #'cash_mov_roll_3',
            #'feature_cash_sig_3',
            #'qty_in',
            #'qty_out',
            #'qty_in_mean',
            #'qty_out_mean',
            #'feature_qty_out_dif',
            #'feature_qty_in_dif',
            #'feature_twrr_non_adjusted',
            #'sd_twrr',
            #'medians',
            #
            #'ratio_m1',
            #'ratio_m2',
            #'ratio_m3',
            #'sum_m3',
            #'sum_m2',
            #'max_m3',
            #'max_m2',
            #'twrr_trends',
            #'cash_trends',
            #'mkt_trends',
            #'medians_trends',
        ]
    }

    training_params = {
        'input_train': data_path('processed', 'features_dataset_train.csv'),
        'input_val': data_path('processed', 'features_dataset_validation.csv'),
        'output': {
            'folders': [
                data_path('modeling', 'output'),
            ],
            'ltb_model': data_path('modeling', 'model.joblib'),
            'xtrain': data_path('modeling', 'output', 'xtrain.csv'),
            'ytrain': data_path('modeling', 'output', 'ytrain.csv'),
            'xval': data_path('modeling', 'output', 'xval.csv'),
            'yval': data_path('modeling', 'output', 'yval.csv'),
        },
        'model': {
            'ltb_params': {
                'verbosity': -1,
                'is_unbalance': True,
                'objective': 'binary',
                'metric': ['auc'],
                'learning_rate': 0.1,
                'boosting_type': 'gbdt',
                'reg_alpha': 0.0,
                'importance_type': 'gain',
                'colsample_bytree': 1.0,
                "n_estimators": 1000,
                "max_depth": 8,
                "num_leaves": 140,
            },
            'threshold': 0.5,
            'nfolds': 10,
        }
    }

