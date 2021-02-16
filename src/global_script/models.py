import math
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import compute_class_weight
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, \
    RepeatVector, TimeDistributed, SimpleRNN
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
import joblib
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dot, LSTM
from tensorflow.keras.models import Model
import cvxpy as cp
from cvxpy.atoms.elementwise.sqrt import sqrt
from cvxpy.atoms.affine.binary_operators import multiply
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall, CategoricalAccuracy

# Prediction function

def prediction(model, NN, x_test, y_test):
    if NN:
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))
    prediction_array = model.predict(x_test)
    prediction_array = prediction_array.reshape(-1, 1)
    pred = pd.DataFrame(prediction_array, index=y_test.index)
    return pred


# Define models

# Neural Nets

def fit_gru(x_train, y_train, x_val):
    tf.keras.backend.clear_session()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
    X_train_array = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_val = x_val[['X(t)']]
    X_val_array = x_val.drop(['X(t)'], axis=1).values.reshape((x_val.shape[0], x_val.drop(['X(t)'], axis=1).shape[1], 1))
    # model
    model = Sequential()
    model.add(GRU(75, return_sequences=True, input_shape=(X_train_array.shape[1], X_train_array.shape[2])))
    model.add(GRU(units=30, return_sequences=True))
    model.add(GRU(units=30))
    model.add(Dense(units=1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit network
    model.fit(X_train_array, y_train, epochs=100, batch_size=128, validation_data=(X_val_array, y_val), shuffle=False, verbose=2,
              callbacks=[es])
    return model


def fit_lstm(x_train, y_train, x_val):
    tf.keras.backend.clear_session()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
    X_train_array = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_val = x_val[['X(t)']]
    X_val_array = x_val.drop(['X(t)'], axis=1).values.reshape((x_val.shape[0], x_val.drop(['X(t)'], axis=1).shape[1], 1))
    # model
    lstm_model = Sequential()
    lstm_model.add(LSTM(75, return_sequences=True, input_shape=(X_train_array.shape[1], X_train_array.shape[2])))
    lstm_model.add(LSTM(units=30, return_sequences=True))
    lstm_model.add(LSTM(units=30))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit network
    lstm_model.fit(X_train_array, y_train, epochs=100, batch_size=128, validation_data=(X_val_array, y_val), shuffle=False, verbose=2,
                   callbacks=[es])
    return lstm_model


def fit_cnn(x_train, y_train, x_val):
    tf.keras.backend.clear_session()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
    X_train_array = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_val = x_val[['X(t)']]
    X_val_array = x_val.drop(['X(t)'], axis=1).values.reshape((x_val.shape[0], x_val.drop(['X(t)'], axis=1).shape[1], 1))
    # define model
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                         input_shape=(X_train_array.shape[1], X_train_array.shape[2])))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(50, activation='relu'))
    model_cnn.add(Dense(1))
    model_cnn.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit network
    model_cnn.fit(X_train_array, y_train, epochs=100, batch_size=128, validation_data=(X_val_array, y_val), shuffle=False, verbose=2,
                  callbacks=[es])
    return model_cnn


def fit_cnn_lstm(x_train, y_train, x_val):
    tf.keras.backend.clear_session()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
    X_train_array = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_val = x_val[['X(t)']]
    X_val_array = x_val.drop(['X(t)'], axis=1).values.reshape((x_val.shape[0], x_val.drop(['X(t)'], axis=1).shape[1], 1))
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=(X_train_array.shape[1], X_train_array.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit network
    model.fit(X_train_array, y_train, epochs=100, batch_size=128, validation_data=(X_val_array, y_val), shuffle=False, verbose=2,
              callbacks=[es])
    return model


def fit_rnn(x_train, y_train, x_val):
    tf.keras.backend.clear_session()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
    X_train_array = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_val = x_val[['X(t)']]
    X_val_array = x_val.drop(['X(t)'], axis=1).values.reshape((x_val.shape[0], x_val.drop(['X(t)'], axis=1).shape[1], 1))
    # model
    model = Sequential()
    model.add(SimpleRNN(64, return_sequences=True, input_shape=(X_train_array.shape[1], X_train_array.shape[2])))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.SimpleRNN(64))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit network
    model.fit(X_train_array, y_train, epochs=100, batch_size=128, validation_data=(X_val_array, y_val), shuffle=False, verbose=2,
              callbacks=[es])
    return model


def fit_fnn(x_train, y_train, x_val):
    tf.keras.backend.clear_session()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
    X_train_array = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_val = x_val[['X(t)']]
    X_val_array = x_val.drop(['X(t)'], axis=1).values.reshape((x_val.shape[0], x_val.drop(['X(t)'], axis=1).shape[1], 1))
    model = Sequential()
    model.add(
        Dense(X_train_array.shape[1], activation='relu', input_shape=(X_train_array.shape[1], X_train_array.shape[2])))
    model.add(Dense(X_train_array.shape[1], activation='relu'))
    model.add(Dense(X_train_array.shape[1], activation='relu'))
    model.add(Dense(X_train_array.shape[1], activation='relu'))
    model.add(Dense(X_train_array.shape[1], activation='relu'))
    model.add(Flatten())
    model.add(Dense(X_train_array.shape[1] // 2, activation='relu'))
    model.add(Dense(1))
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=nadam, loss='mae', metrics=['mse'])
    # fit network
    model.fit(X_train_array, y_train, epochs=100, batch_size=128, validation_data=(X_val_array, y_val), shuffle=False, verbose=2,
              callbacks=[es])
    return model


# Booster Algorithms

def fit_xgboost(x_train, y_train, n):
    params = {"learning_rate": [0.05, 0.30, 0.50],
              "max_depth": [10, 15],
              "min_child_weight": [1, 7],
              "gamma": [0.0, 0.2],
              "colsample_bytree": [0.7]}
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    # model
    boost_model = RandomizedSearchCV(XGBRegressor(n_estimators=n), params, scoring=mse, n_jobs=-1, cv=5, verbose=1, n_iter=5)
    best_model = boost_model.fit(x_train.values, y_train.values, eval_metric='mae')
    return best_model.best_estimator_


def fit_lgb(x_train, y_train):
    # model
    model = LGBMRegressor(n_estimators=1000)
    model.fit(x_train, y_train, eval_metric='mae')
    return model


# Classic models

"""def fit_svr(x_train, y_train, kernel, path, horizon, hour, k):
    # model
    C = max(np.abs(y_train.values.mean() - 3 * y_train.values.std()),
            np.abs(y_train.values.mean() + 3 * y_train.values.std()))
    C = max(C, 0.1)
    knn_model = joblib.load(os.path.join(path, str("KNN_" + str(horizon) + '_' + str(hour) + ".pkl")))
    knn_pred = knn_model.predict(x_train)
    n = y_train.shape[0]
    var_noise = pow(n, 1 / 5) * k / (pow(n, 1 / 5) - 1) * 1 / n * np.sum(np.square(np.subtract(y_train, knn_pred)))
    eps = 3 * np.sqrt(var_noise * np.log(n) / n)
    model = SVR(kernel=kernel, C=C, epsilon=eps, gamma="auto")
    model.fit(x_train, y_train)
    return model"""

def fit_svr(x_train, y_train, knn_model, k):
    # model
    C = max(np.abs(y_train.values.mean() - 3 * y_train.values.std()),
            np.abs(y_train.values.mean() + 3 * y_train.values.std()))
    C = max(C, 0.1)
    #knn_model = joblib.load(os.path.join(path, str("KNN_" + str(horizon) + '_' + str(hour) + ".pkl")))
    knn_pred = knn_model.predict(x_train)
    n = y_train.shape[0]
    var_noise = pow(n, 1 / 5) * k / (pow(n, 1 / 5) - 1) * 1 / n * np.sum(np.square(np.subtract(y_train, knn_pred)))
    eps = 3 * np.sqrt(var_noise * np.log(n) / n)
    model = SVR(kernel='rbf', C=C, epsilon=eps, gamma="auto")
    model.fit(x_train, y_train)
    return model

def fit_lr(x_train, y_train):
    # model
    model = LinearRegression().fit(x_train, y_train)
    return model


def fit_knn(x_train, y_train):
    n_neighbors = list(range(int(x_train.shape[0]/10), int(x_train.shape[0]/2)))
    #n_neighbors = list(range(1, 10))
    hyperparameters = dict(n_neighbors=n_neighbors)
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    clf = RandomizedSearchCV(KNeighborsRegressor(), hyperparameters, scoring=mse, n_jobs=-1, cv=5, verbose=1, n_iter=10)
    model = clf.fit(x_train, y_train)
    k = model.best_estimator_.get_params()['n_neighbors']
    return model.best_estimator_, k


def fit_rf(x_train, y_train):
    model = RandomForestRegressor().fit(x_train, y_train)
    return model

# Combine models

def fit_COP(y_ens, y_experts, type):
    print("TYPE", type)
    M = y_experts.shape[1]
    T = y_experts.shape[0]
    W = cp.Variable(M)
    one = np.ones(M)
    y = y_ens.values.reshape(T)
    y_hat = y_experts.values
    if type == 'none':
        cost = sqrt(cp.sum_squares((y - y_hat @ W) )* 1 / T)
        prob = cp.Problem(cp.Minimize(cost), [W >= 0, one.T @ W == 1])
    elif type == 'inverse' or type=='exp':
        rmse_ = error(y_ens, y_experts, rmse)
        nrmse_ = error(y_ens, y_experts, nrmse)
        if type == 'inverse':
            error_ = [1 / (x + y) for x, y in zip(rmse_,  nrmse_)]
        if type == 'exp':
            error_ = [np.exp(-(x+y)) for x, y in zip(rmse_,  nrmse_)]
        cost = sqrt(cp.sum_squares(y - y_hat @ multiply(W, error_)) * 1 / T)
        prob = cp.Problem(cp.Minimize(cost), [multiply(W, error_) >= 0, one.T @ multiply(W, error_) == 1])
    else :
        rmse_ = error(y_ens, y_experts, rmse)
        nrmse_ = error(y_ens, y_experts, nrmse)
        cost = sqrt(cp.sum_squares(y - y_hat @ (W - rmse_ - nrmse_)) * 1 / T)
        prob = cp.Problem(cp.Minimize(cost), [W - rmse_ - nrmse_ >= 0, one.T @ (W - rmse_ - nrmse_) == 1])
    try:
        prob.solve(qcp=True, solver=cp.XPRESS)
    except:
        try:
            prob.solve(qcp=True, solver=cp.CVXOPT)
        except:
            try:
                prob.solve(qcp=True, solver=cp.ECOS)
            except:
                print('Nothing works')
                return(np.ones(M)/M)
    print("Infeasible", prob.value == np.inf)
    if type == 'none':
        return(W.value)
    elif type == 'inverse' or type=='exp':
        if (prob.value == np.inf):
            return(error_)
        else :
            return np.multiply(W.value, error_)
    else :
        return(W.value - rmse_ - nrmse_)

#MOE
def fit_MOE(x_ens, y_experts, y_ens):
    y_experts = y_experts.reindex(x_ens.index)
    x_ens_array = x_ens.values.reshape((x_ens.shape[0], x_ens.shape[1], 1))
    y_experts_array = y_experts.values.reshape((y_experts.shape[0], y_experts.shape[1]))
    neurons_lstm = 64
    neurons_reduction = 8

    gater = Input(shape=(x_ens_array.shape[1], 1))
    lstm = LSTM(neurons_lstm)(gater)
    G = Dense(neurons_reduction, activation='softmax')(lstm)

    experts = Input(shape=(y_experts_array.shape[1],))
    reduction = Dense(neurons_reduction, activation='relu')(experts)

    merge = Dot(axes=(1))([reduction, G])
    output = Dense(1)(merge)
    model = Model([gater, experts], outputs=output)

    model.compile(optimizer='Adamax', loss='mae', metrics=['mse'])
    model.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=3)
    model.fit([x_ens_array, y_experts], y_ens, epochs=200, batch_size=10, validation_split=0.2, shuffle=True, verbose=1,
              callbacks=[es])
    return model

# Error

def error(y, df, function):
    error_ = []
    for model in df.columns:
        error_.append(function(y, df[model]))
    if function == nrmse:
        error_/=np.mean(error_)
    return (error_)

def nrmse(y, pred):
    y[y == 0] = np.finfo(float).eps
    e = np.mean(np.sqrt(1/len(y)*np.sum(np.square(np.divide(np.subtract(y.values,pred.values), y.values)))))
    return e

def rmse(y, pred) :
    return(mean_squared_error(y, pred, squared = False))

# Neural Net classifier for PSF algorithm
def classifier(X, Y, clusters):

    X_train = X.sample(frac=0.8)
    Y_train = Y.loc[X_train.index]
    X_val = X.drop(X_train.index)
    Y_val = Y.drop(X_train.index)

    c_w = compute_class_weight('balanced', np.unique(clusters), clusters)
    c_w = dict(enumerate(c_w))

    METRICS = [
        Recall(name='recall'),
        AUC(name='auc', multi_label=False)
    ]

    es = EarlyStopping(monitor='weighted_recall', mode='max', verbose=0, patience=6)
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS, weighted_metrics=METRICS)
    model.fit(X_train, Y_train, epochs=500, batch_size=128, validation_data=(X_val, Y_val),
              shuffle=False, verbose=1, callbacks=[es], class_weight=c_w)
    return model