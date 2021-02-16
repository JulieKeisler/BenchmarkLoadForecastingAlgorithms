import global_script.models as mod
import tensorflow as tf
import os

# General function which fit all models on the training dataset
def fit_all_models(df, c):
    X = df.drop(['X(t)'], axis=1)
    Y = df[['X(t)']]
    X_train = X.sample(frac=0.9, random_state=200)
    Y_train = Y.loc[X_train.index]
    X_val = df.drop(X_train.index)
    models = {}
    m_knn, k = mod.fit_knn(X_train, Y_train)
    models['KNN'] = m_knn
    m = mod.fit_svr(X_train, Y_train, m_knn, k)
    models['SVR'] = m

    models_NN = {"GRU": mod.fit_gru, "CNN_LSTM": mod.fit_cnn_lstm, "CNN": mod.fit_cnn, "LSTM": mod.fit_lstm,
                 "RNN": mod.fit_rnn, "FNN": mod.fit_fnn}
    for model in models_NN:
        m = models_NN[model](X_train, Y_train, X_val)
        m.save(os.path.join("Model/PSF_DTW", str(model + "_" + str(c) + ".h5")))
        models[model] = m

    models_ = {"LGB": mod.fit_lgb, "LR": mod.fit_lr, "RF": mod.fit_rf}
    for model in models_:
        m = models_[model](X_train, Y_train)
        models[model] = m

    m = mod.fit_xgboost(X_train, Y_train, 1000)
    models['XGB'] = m
    return models

# fit Convex Optimisation Problem
def fit_all_COP(df, df_experts) :
    # COP
    Y_ens = df[['X(t)']]
    models = {}
    error = ["none", "exp", "inverse", "sum"]
    for e in error:
        W_final = mod.fit_COP(Y_ens, df_experts, e)
        models['COP_' + e] = W_final
    return(models)

# Compute prediction of model on the samples of df belonging to cluster c
def prediction(model, df, name, c):
    models_NN = ["GRU", "CNN_LSTM", "CNN", "LSTM", "RNN", "FNN"]
    models_standard = ["LGB", "LR", "RF", "SVR", "KNN"]
    COP_models = ["COP_none", "COP_exp", "COP_inverse", "COP_sum"]

    if name not in COP_models:
        if len(df.shape)>1:
            x_test = df.drop(['X(t)'], axis=1)
        else:
            x_test = df.drop(['X(t)'])
    if name.strip() in models_NN:
        m = tf.keras.models.load_model(os.path.join("Model/PSF_DTW", str(name + "_" + str(c) + ".h5")))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))
        prediction_array = m.predict(x_test)
        pred = prediction_array
    elif name.strip() in models_standard:
        prediction_array = model.predict(x_test)
        pred = prediction_array
    elif name.strip() == "XGB":
        prediction_array = model.predict(x_test.values)
        pred = prediction_array
    elif name.strip() in COP_models :
        pred = df @ model
    pred = pred.ravel()
    return(pred)


