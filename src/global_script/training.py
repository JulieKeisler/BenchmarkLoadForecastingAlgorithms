import pandas as pd
import MEGA_SCRIPT_models as mod
import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import tensorflow as tf

#train all individual learners with or without clusters
def train_all_models(dtrain, dens, dtest, cluster, h):
    X_train = dtrain.copy()
    X_test = dtest.copy()
    X_ens = dens.copy()
    tf.keras.backend.clear_session()
    repos_path = "Data"
    model_path = "Model/PSF"
    Y_train = X_train[['X(t)']]
    X_val = X_ens.sample(X_ens.shape[0]//10)
    Y_ens = X_ens[['X(t)']]
    Y_test = X_test[['X(t)']]
    X_train.drop(['X(t)'], axis=1, inplace=True)
    X_ens.drop(['X(t)'], axis=1, inplace=True)
    X_test.drop(['X(t)'], axis=1, inplace=True)
    # fit and save models
    y_hat_ens = pd.DataFrame(index=Y_ens.index)
    y_hat_test = pd.DataFrame(index=Y_test.index)
    print(X_train.head())

    model = "KNN"
    m, k = mod.fit_knn(X_train, Y_train)
    joblib.dump(m, os.path.join(model_path, str(model + "_" + str(h) + '_' + str(cluster) + ".pkl")))
    y_hat_test[model] = mod.prediction(m, False, X_test, Y_test)
    y_hat_ens[model] = mod.prediction(m, False, X_ens, Y_ens)

    models_SVR = ["SVR"]
    kernels = ['rbf']
    for model in models_SVR:
        for kernel in kernels:
            m = mod.fit_svr(X_train, Y_train, kernel, model_path, h, cluster, k)
            y_hat_test[model] = mod.prediction(m, False, X_test, Y_test)
            y_hat_ens[model] = mod.prediction(m, False, X_ens, Y_ens)
            del m

    models_NN = {"CNN_LSTM": mod.fit_cnn_lstm, "CNN": mod.fit_cnn, "GRU": mod.fit_gru, "LSTM": mod.fit_lstm,
                 "RNN": mod.fit_rnn,"FNN": mod.fit_fnn}

    models = {"LGB": mod.fit_lgb, "LR": mod.fit_lr, "RF": mod.fit_rf}

    for model in models:
        m = models[model](X_train, Y_train)
        y_hat_test[model] = mod.prediction(m, False, X_test, Y_test)
        y_hat_ens[model] = mod.prediction(m, False, X_ens, Y_ens)
        del m

    for model in models_NN:
        m = models_NN[model](X_train, Y_train, X_val)
        y_hat_test[model] = mod.prediction(m, True, X_test, Y_test)
        y_hat_ens[model] = mod.prediction(m, True, X_ens, Y_ens)
        del m

    model = "XGB"
    for n in [1000]:
        m = mod.fit_xgboost(X_train, Y_train, n)
        y_hat_test[model] = mod.prediction(m, False, X_test, Y_test)
        y_hat_ens[model] = mod.prediction(m, False, X_ens, Y_ens)
        del m

    if cluster != "global":
        # COP
        error = ["none", "exp", "inverse", "sum"]
        cop_columns = []
        for e in error :
            W_final = mod.fit_COP(Y_ens, y_hat_ens, e)
            np.save(os.path.join(model_path, str("COP_W_" + str(h) + '_' + str(cluster) + "_" + e + ".pkl")), W_final)
            COP_prediction = y_hat_test.drop(cop_columns, axis=1) @ W_final
            y_hat_test['COP_'+e] = COP_prediction
            cop_columns.append('COP_'+e)

    return (y_hat_test, y_hat_ens)

# create features for individual learners
def create_ts_data(df_base, lags, forecast):
    df_ = df_base.copy()
    if 'cluster' in df_.columns:
        df_weather = df_[['Temperature', 'Precipitation', 'cluster']]
        df_.drop(['Temperature', 'Precipitation', 'cluster'], axis=1, inplace=True)
    else :
        df_weather = df_[['Temperature', 'Precipitation']]
        df_.drop(['Temperature', 'Precipitation'], axis=1, inplace=True)
    # Historical features
    cols, names = list(), list()
    for i in lags:
        cols.append(df_.shift(i))
        names += ['X(t-%d)' % i]
    # Calendar features
    cols_time = list()
    cols_time.append(pd.DataFrame(df_.index.hour.values))
    cols_time.append(pd.DataFrame(df_.index.weekday.values))
    cols_time.append(pd.DataFrame(df_.index.month.values))
    names_time = ['H', 'day', 'month']
    # Forecast value
    cols.append(df_[forecast])
    cols.append(df_weather['Temperature'])
    if 'cluster' in df_weather.columns:
        cols.append(df_weather['cluster'])
        names += ['X(t)', 'temperature(t)', 'cluster',]
    else:
        names += ['X(t)', 'temperature(t)']
    # Concatenate calendar features and convert to dummies
    agg_time = pd.concat(cols_time, axis=1).reset_index(drop=True)
    agg_time.columns = names_time
    agg_time = pd.get_dummies(agg_time, columns=names_time)
    # Concatenate forecast and historical features
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg_wo_index = agg.copy().reset_index(drop=True)
    agg.dropna(inplace=True)
    # Concatenate calendar features with the others
    agg_wo_index = pd.concat((agg_time, agg_wo_index), axis=1)
    # drop rows with NaN values
    agg_wo_index.dropna(inplace=True)
    agg_wo_index.set_index(agg.index, inplace=True)
    return agg_wo_index

