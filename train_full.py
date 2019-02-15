import random
import os
import numpy as np
import pandas as pd
import utils
import catboost
from catboost import Pool
from sklearn.model_selection import KFold, train_test_split
import scoring
import xgboost as xgb
import lightgbm as lgb
import pickle


random.seed(42)
print('start')
pd.options.mode.chained_assignment = None


def read_data():
    data_path = "../IDAO-MuID"
    train_concat = pd.read_hdf('train_concat_31.01_rmse_foi.h5')
    full_train = utils.load_train_hdf(data_path)
    print('read')
    y_source = full_train.label
    weight_source = full_train.weight

    neg = full_train.weight < 0
    full_train.label[neg] = (full_train.label[neg] + 1) % 2
    abs_weights = np.abs(full_train.weight)
    abs_weights = abs_weights / np.max(abs_weights)
    return train_concat, full_train.label, y_source, abs_weights, weight_source


def train_xgb(train_data: pd.DataFrame,
              labels: pd.DataFrame,
              labels_source: pd.DataFrame,
              abs_weights: pd.DataFrame,
              source_weights: pd.DataFrame,
              n_est=1000,
              verbose=False):

    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=1)
    d = 7

    for fold_n, (train_index, test_index) in enumerate(folds.split(train_data)):
        print(f'start fold {fold_n}')
        x_train, x_val = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_val = labels.iloc[train_index], labels.iloc[test_index]
        y_train_source, y_val_source = labels_source.iloc[train_index], labels_source.iloc[test_index]
        w_train, w_val = abs_weights.iloc[train_index], abs_weights.iloc[test_index]
        w_train_source, w_val_source = source_weights.iloc[train_index], source_weights.iloc[test_index]

        xgb_classifier = xgb.XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=n_est, n_jobs=30,
                                           silent=True)

        xgb_classifier.fit(x_train.values, y_train.values, sample_weight=w_train.values,
                           eval_set=[(x_train.values, y_train.values), (x_val.values, y_val.values)], verbose=verbose)

        validation_predictions = xgb_classifier.predict_proba(x_val.values)
        curr_score = scoring.rejection90(y_val_source.values, validation_predictions[:, 1], sample_weight=w_val_source.values)
        print(f'for fold_n {fold_n}, d {d} score is {curr_score:.3f}')
        with open(f'models_result/xgb_{fold_n}.pkl', 'wb') as f_out:
            pickle.dump(xgb_classifier, f_out)


def train_lgb(train_data: pd.DataFrame,
              labels: pd.DataFrame,
              labels_source: pd.DataFrame,
              abs_weights: pd.DataFrame,
              source_weights: pd.DataFrame,
              n_est=1000,
              verbose=False):
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=1)
    d = 7

    for fold_n, (train_index, test_index) in enumerate(folds.split(train_data)):
        print(f'start fold {fold_n}')
        x_train, x_val = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_val = labels.iloc[train_index], labels.iloc[test_index]
        y_train_source, y_val_source = labels_source.iloc[train_index], labels_source.iloc[test_index]
        w_train, w_val = abs_weights.iloc[train_index], abs_weights.iloc[test_index]
        w_train_source, w_val_source = source_weights.iloc[train_index], source_weights.iloc[test_index]

        gbm = lgb.LGBMClassifier(learning_rate=0.1, objective='binary', feature_fraction=0.8,
                                 bagging_fraction=0.8, bagging_freq=1, n_estimators=n_est, max_depth=d, reg_lambda=3,
                                 num_leaves=d * 3, n_jobs=30, silent=True)
        gbm.fit(x_train, y_train, sample_weight=w_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=verbose)

        validation_predictions = gbm.predict_proba(x_val)
        curr_score = scoring.rejection90(y_val_source.values, validation_predictions[:, 1],
                                         sample_weight=w_val_source.values)
        print(f'for fold_n {fold_n}, d {d} score is {curr_score:.3f}')
        with open(f'models_result/lgb_{fold_n}.pkl', 'wb') as f_out:
            pickle.dump(gbm, f_out)


def train_catboost(train_data: pd.DataFrame,
                   labels: pd.DataFrame,
                   abs_weights: pd.DataFrame,
                   n_est=2500,
                   verbose=False):
    for index, d in enumerate([6, 7, 7, 7, 8]):
        print(f'start train catboost with d = {d}, index = {index}')
        x_train, _, y_train, _, w_train, _ = train_test_split(train_data, labels, abs_weights, test_size=0.05)
        model = catboost.CatBoostClassifier(iterations=n_est, max_depth=d, thread_count=30, verbose=verbose)
        model.fit(x_train, y_train, sample_weight=w_train, plot=False)
        model.save_model(f"models_result/catboost_{index}.cbm")


if __name__ == '__main__':
    train_data_full, train_labels, source_train_labels, train_abs_weights, train_weight_source = read_data()
    train_lgb(train_data_full, train_labels, source_train_labels, train_abs_weights, train_weight_source)
    train_xgb(train_data_full, train_labels, source_train_labels, train_abs_weights, train_weight_source)
    train_catboost(train_data_full, train_labels, train_abs_weights)
