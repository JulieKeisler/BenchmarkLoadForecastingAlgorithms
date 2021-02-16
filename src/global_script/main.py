## import
import pandas as pd
import os
import clustering as cluster
import classification as clf
import training as training
import models as mod
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math


## Scenario function
def scenario(dataset, horizons, clustering_method):
    # load data
    data = cluster.clustering_type(clustering_method, dataset)
    data.to_pickle(os.path.join("Data/results/UCI_Hourly/MEGA", str(clustering_method + "_clustered.pkl")))
    for horizon in horizons:
        # Associate clusters to each sample
        dtrain, dtest = clf.train_pred_classifier(data, horizon, clustering_method)
        data = pd.concat([dtrain, dtest], axis=0)
        # Add weather data
        weather_filename = os.path.join(repos_path, "weather_uci.csv")
        df_weather = pd.read_csv(weather_filename, parse_dates=True, infer_datetime_format=True, index_col=0, sep=',')
        data = pd.merge(data, df_weather, left_index=True, right_index=True)
        # Create features for individual learners
        k = math.ceil(horizon / 12)
        lags = [horizon, horizon + 1, horizon + 2, horizon + 3, 12 * k - 1, 12 * k, 12 * k + 1, 12 * (k + 1), 12 * (k + 2), 12 * (k + 3)]
        lags = list(set(lags))
        lags.sort()
        data = training.create_ts_data(data, lags, 'X(t)')
        dtrain = data['2006':'2008']
        dtest = data['2009':'2010'
        # Scale Data
        min_y = dtrain['X(t)'].min(axis=0)
        max_y = dtrain['X(t)'].max(axis=0)
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        num_cols = dtrain.columns[dtrain.dtypes == float]
        dtrain[list(num_cols)] = scaler_x.fit_transform(dtrain[list(num_cols)])
        dtest[list(num_cols)] = scaler_x.transform(dtest[list(num_cols)])

        clusters = pd.unique(dtrain['cluster'])
        clusters = clusters[~np.isnan(clusters)]
        clusters_test = []
        clusters_ens = []
        dens = dtest['2009']
        dtest = dtest['2010']
        # Train and predict models for each cluster
        for c in clusters :
            Y_hat_test_clustered, Y_hat_ens_clustered = \
                training.train_all_models(dtrain[dtrain["cluster"]==c], dens[dens["cluster"]==c], dtest[dtest["cluster"]==c], str(clustering_method + str(c)), horizon)
            clusters_test.append(Y_hat_test_clustered)
            clusters_ens.append(Y_hat_ens_clustered)
        # Concatenate clusters predictions
        prediction_test_clusters = pd.DataFrame(index=dtest.index)
        columns = clusters_test[0].columns
        prediction_test_clusters[columns] = clusters_test[0]
        for c in range(1, len(clusters)):
            prediction_test_clusters[columns] = prediction_test_clusters[columns].fillna(clusters_test[c])
        prediction_ens_clusters = pd.DataFrame(index=dens.index)
        columns = clusters_ens[0].columns
        prediction_ens_clusters[columns] = clusters_ens[0]
        for c in range(1, len(clusters)):
            prediction_ens_clusters[columns] = prediction_ens_clusters[columns].fillna(clusters_ens[c])
        dtrain.drop(['cluster'], axis=1, inplace=True)
        dens.drop(['cluster'], axis=1, inplace=True)
        dtest.drop(['cluster'], axis=1, inplace=True)
        # Train and predict models on the whole dataset (without the notion of clusters)
        pred_global_test, pred_global_ens = training.train_all_models(dtrain, dens, dtest, 'global', horizon)
        best_ens_prediction = pd.DataFrame(index=dens.index)
        best_test_prediction = pd.DataFrame(index=dtest.index)
        sys.stdout.write("Choose best model")
        # For each models choose between the model trained on the whole dataset and the one trained on each cluster
        for model in columns:
            rmse_clusters = mod.rmse(dens[['X(t)']], prediction_ens_clusters[model])
            nrmse_clusters = mod.nrmse(dens[['X(t)']], prediction_ens_clusters[model])
            rmse_global = mod.rmse(dens[['X(t)']], pred_global_ens[model])
            nrmse_global = mod.nrmse(dens[['X(t)']], pred_global_ens[model])
            model_index = np.argmin([rmse_global + nrmse_global, rmse_clusters + nrmse_clusters])
            if model_index == 1:
                best_ens_prediction[model] = prediction_ens_clusters[model]
                best_test_prediction[model] = prediction_test_clusters[model]
            else:
                best_ens_prediction[model] = pred_global_ens[model]
                best_test_prediction[model] = pred_global_test[model]

        for col in best_test_prediction:
            best_test_prediction.rename(columns={col: col+"_best"}, inplace=True)
        for col in prediction_test_clusters:
            prediction_test_clusters.rename(columns={col: col+"_cluster"}, inplace=True)
        for col in pred_global_test:
            pred_global_test.rename(columns={col: col+"_global"}, inplace=True)

        # MOE
        cop_columns = [col for col in prediction_test_clusters if col.startswith('COP')]
        all_prediction_ens = pd.concat([pred_global_ens, prediction_ens_clusters],axis=1)
        data_ens = {"best" : best_ens_prediction, "all" : all_prediction_ens, "global" : pred_global_ens, "clusters" : prediction_ens_clusters}
        all_prediction_test = pd.concat([pred_global_test, prediction_test_clusters], axis=1)
        data_test = {"best": best_test_prediction, "all": all_prediction_test.drop(cop_columns, axis=1), "global": pred_global_test,
                    "clusters": prediction_test_clusters.drop(cop_columns, axis=1)}
        moe_test = pd.DataFrame(index=dtest.index)
        for data in data_ens:
            data_ens[data].to_pickle(os.path.join("Data/results/UCI_Hourly/MEGA", str(clustering_method + "_" + str(horizon) + "_ens_" + data + ".pkl")))
            data_test[data].to_pickle(os.path.join("Data/results/UCI_Hourly/MEGA", str(clustering_method + "_" + str(horizon) + "_test_" + data + ".pkl")))
            m = mod.fit_MOE(dens.drop(['X(t)'], axis=1), data_ens[data], dens[['X(t)']])
            X_test_array = dtest.drop(['X(t)'], axis=1).values.reshape((dtest.drop(['X(t)'], axis=1).shape[0], dtest.drop(['X(t)'], axis=1).shape[1], 1))
            y_hat_test_array = data_test[data].values.reshape((data_test[data].shape[0], data_test[data].shape[1]))
            MOE_prediction = m.predict([X_test_array, y_hat_test_array])
            MOE_prediction = MOE_prediction.reshape(-1, 1)
            MOE_prediction = pd.DataFrame(MOE_prediction, index=dtest.index)

            moe_test['MOE_' + data] = MOE_prediction

        best_test_prediction['AVG'] = np.mean(best_test_prediction, axis=1)

        best_test_prediction = best_test_prediction * (max_y - min_y) + min_y
        best_test_prediction[best_test_prediction < 0] = 0
        best_test_prediction[best_test_prediction > max_y] = max_y
        prediction_test_clusters = prediction_test_clusters * (max_y - min_y) + min_y
        prediction_test_clusters[prediction_test_clusters < 0] = 0
        prediction_test_clusters[prediction_test_clusters > max_y] = max_y
        pred_global_test = pred_global_test * (max_y - min_y) + min_y
        pred_global_test[pred_global_test < 0] = 0
        pred_global_test[pred_global_test > max_y] = max_y
        moe_test = moe_test * (max_y - min_y) + min_y
        moe_test[moe_test < 0] = 0
        moe_test[moe_test > max_y] = max_y

        prediction = pd.concat([best_test_prediction, prediction_test_clusters, pred_global_test, moe_test], axis=1)
        prediction.to_pickle(
            os.path.join(repos_path, "results", "UCI_Hourly/MEGA", str(str(horizon) + "_" + clustering_method + "_prediction.pkl")))

## Variables
h = [1,4,24,48,168] # Forecast horizons
repos_path = "Data"
model_path = "Model/MEGA"


## Clustering methods

clustering = ['hourly', 'dtw_24', 'dtw_4', 'euclidean']


for clustering_method in clustering :
    ## Load Data
    if clustering_method == 'hourly' or clustering_method=='euclidean_24' or clustering_methode=='euclidean_4':
        data = pd.read_csv(os.path.join(repos_path, "y.csv"), parse_dates=True, infer_datetime_format=True, index_col=0, sep=',')
        data.drop(['clusters'], axis=1, inplace=True)
    elif clustering_method == 'dtw_24':
        data = pd.read_csv(os.path.join(repos_path, "data_with_cluster_4.csv"), parse_dates=True, infer_datetime_format=True, index_col=0, sep=',')
    elif clustering_method == 'dtw_4':
        data = pd.read_csv(os.path.join(repos_path, "data_with_cluster_24.csv"), parse_dates=True, infer_datetime_format=True, index_col=0, sep=',')
    scenario(data, h, clustering_method)


