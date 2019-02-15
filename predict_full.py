import os
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error as mse
import utils


# test_concat = pd.read_hdf('test_concat_31.01_rmse_foi.h5')
test_concat = pd.read_hdf('test_private.h5')
data_path = "../IDAO-MuID"
test = utils.load_full_test_csv(data_path)
models_dir = 'models_result'
models = [os.path.join(models_dir, x) for x in os.listdir(models_dir)]
predictions_all = []
for model_name in models:
    print(model_name)
    short_model_name = os.path.split(model_name)[1].split('.')[0]
    if 'xgb' in model_name or 'lgb' in model_name:
        
        with open(model_name, 'rb') as f_in:
            model = pickle.load(f_in)
            predictions = model.predict_proba(test_concat.values)
            predictions = predictions[:, 1]
            predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            prediction_pd = pd.DataFrame(data={"prediction": predictions}, index=test.index)     
            prediction_pd.to_csv(f'predictions/{short_model_name}.csv')
            predictions_all.append(('xgb' if 'xgb' in model_name else 'lgb', prediction_pd))
    else:
        model = CatBoostClassifier()
        model.load_model(model_name)
        predictions = model.predict(test_concat.values, prediction_type="RawFormulaVal").astype(np.float32)
        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
        prediction_pd = pd.DataFrame(data={"prediction": predictions}, index=test_concat.index)

        prediction_pd.to_csv(f'predictions/{short_model_name}.csv')
        predictions_all.append(('catboost', prediction_pd))

avg_xgb = None
avg_lgb = None
avg_catboost = None
for type_, prediction in predictions_all:
    if type_ == 'xgb':
        if avg_xgb is None:
            avg_xgb = prediction
        else:
            avg_xgb.prediction = avg_xgb.prediction + prediction.prediction
    elif type_ == 'lgb':
        if avg_lgb is None:
            avg_lgb = prediction
        else:
            avg_lgb.prediction = avg_lgb.prediction + prediction.prediction
    elif type_ == 'catboost':
        if avg_catboost is None:
            avg_catboost = prediction
        else:
            avg_catboost.prediction = avg_catboost.prediction + prediction.prediction
    else:
        print('ALERT!!!!')

        
        
def convert(solution):
    a_arg = solution.copy()
    s = np.argsort(a_arg)
    for i, value in enumerate(s):
        a_arg[value] = i
    return a_arg
    
    
avg_xgb.prediction = avg_xgb.prediction / len([x for x in predictions_all if x[0] == 'xgb'])
avg_lgb.prediction = avg_lgb.prediction / len([x for x in predictions_all if x[0] == 'lgb'])
avg_catboost.prediction = avg_catboost.prediction / len([x for x in predictions_all if x[0] == 'catboost'])

avg_xgb.prediction = convert(avg_xgb.prediction)
avg_lgb.prediction = convert(avg_lgb.prediction)
avg_catboost.prediction = convert(avg_catboost.prediction)

print(f'xgb max {avg_xgb.max()} min {avg_xgb.min()}')
print(f'lgb max {avg_lgb.max()} min {avg_lgb.min()}')
print(f'catboost max {avg_catboost.max()} min {avg_catboost.min()}')
print(f'mse catboost xgb {mse(avg_catboost.prediction, avg_xgb.prediction)}')
print(f'mse catboost lgb {mse(avg_catboost.prediction, avg_lgb.prediction)}')
print(f'mse lgb xgb {mse(avg_lgb.prediction, avg_xgb.prediction)}')
result = avg_xgb.copy()
result.prediction = (result.prediction*2 + avg_lgb.prediction + avg_catboost.prediction) / 4
result.prediction = (result.prediction - result.prediction.min()) / (result.prediction.max() - result.prediction.min())
avg_xgb.prediction = (avg_xgb.prediction - avg_xgb.prediction.min()) / (avg_xgb.prediction.max() - avg_xgb.prediction.min())
avg_lgb.prediction = (avg_lgb.prediction - avg_lgb.prediction.min()) / (avg_lgb.prediction.max() - avg_lgb.prediction.min())
avg_catboost.prediction = (avg_catboost.prediction - avg_catboost.prediction.min()) / (avg_catboost.prediction.max() - avg_catboost.prediction.min())
print(f'mse result xgb {mse(result.prediction, avg_xgb.prediction)}')
print(f'mse result lgb {mse(result.prediction, avg_lgb.prediction)}')
print(f'mse result catboost {mse(result.prediction, avg_catboost.prediction)}')
result.to_csv(f'predictions/result.csv')
avg_xgb.to_csv(f'predictions/avg_xgb.csv')
avg_lgb.to_csv(f'predictions/avg_lgb.csv')
avg_catboost.to_csv(f'predictions/avg_catboost.csv')