from tslearn.metrics import cdist_dtw
from scipy.cluster.hierarchy import complete, fcluster
import pandas as pd
import os
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import training_psf_dtw as tpd
import global_script.models as mod
import math
import warnings
warnings.filterwarnings("ignore")


# Create features from the dataset for the individual experts
def create_ts_data(df_base, lags, forecast):
    df_ = df_base.copy()
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

#Variables
# forecast horizon
h=4
# Window to determine how many clusters from the past we give to the classifier to predict new clusters
W=42
# Nb of experts for the Mixture of experts
nb_experts = 15
#Nb of clusters
nb_cluster = 6

#Load Data
repos_path = "Data"
data = pd.read_csv(os.path.join(repos_path, "y.csv"), parse_dates=True, infer_datetime_format=True, index_col=0, sep=',')
data.drop(['clusters'], axis=1, inplace=True)

#Load old forecasts, with models trained on the whole dataset without clustering
global_predicted_ens = pd.read_pickle(os.path.join(repos_path, str(h) + "_ens_global.pkl"))
global_predicted_test = pd.read_pickle(os.path.join(repos_path, str(h) + "_test_global.pkl"))
for col in global_predicted_ens.columns:
    global_predicted_ens.rename(columns={col: col + "_global"}, inplace=True)

#Load weather dataset
weather_filename = os.path.join(repos_path, "weather_uci.csv")
df_weather = pd.read_csv(weather_filename, parse_dates=True, infer_datetime_format=True, index_col=0, sep=',')
data = pd.merge(data, df_weather, left_index=True, right_index=True)

#Define supervised problem by creating the features for expert models
horizon=h
k = math.ceil(horizon / 12)
lags = [horizon, horizon + 1, horizon + 2, horizon + 3, 12 * k - 1, 12 * k, 12 * k + 1, 12 * (k + 1), 12 * (k + 2), 12 * (k + 3)]
lags = list(set(lags))
lags.sort()
data = create_ts_data(data, lags, 'X(t)')
dtrain = data['2006':'2008']
dens = data['2009']
dtest = data['2010']

#Scale data
min_y = dtrain['X(t)'].min(axis=0)
max_y = dtrain['X(t)'].max(axis=0)
scaler_x = MinMaxScaler(feature_range=(0, 1))
num_cols_d = dtrain.columns[dtrain.dtypes == float]
dtrain[list(num_cols_d)] = scaler_x.fit_transform(dtrain[list(num_cols_d)])
dens[list(num_cols_d)] = scaler_x.fit_transform(dens[list(num_cols_d)])
dtest[list(num_cols_d)] = scaler_x.transform(dtest[list(num_cols_d)])
"""for model in global_predicted_test.columns:
    global_predicted_ens[model] = (global_predicted_ens[model]-min_y)/(max_y-min_y)
    global_predicted_test[model] = (global_predicted_test[model]-min_y)/(max_y-min_y)"""
dtest.dropna(inplace=True)

#Reshape original data by subsequences of four hours
data_day = data[['X(t)']]
conditions = [
    (data_day.index.hour < 4),
    (data_day.index.hour >= 4) & (data_day.index.hour < 8),
    (data_day.index.hour >= 8) & (data_day.index.hour < 12),
    (data_day.index.hour >= 12) & (data_day.index.hour < 16),
    (data_day.index.hour >= 16) & (data_day.index.hour < 20),
    (data_day.index.hour >= 20)
    ]
values = [" 0" + str(k*4) + ":00" for k in range(6)]
data_day['categories'] =np.select(conditions, values)
data_day['date'] = data_day.index.date.astype(str) + data_day['categories'].astype(str)
data_day['hour'] = data_day.index.hour % 4
data_day = data_day.pivot_table(index='date', columns='hour', values='X(t)')
data_day.dropna(inplace=True)
data_day.index = pd.to_datetime(data_day.index).sort_values()
dtrain_day = data_day['2006':'2008']
dens_day = data_day['2009']
dtest_day = data_day['2010']

#Cluster training dataset using DTW and linkage
print("CLustering...")
distance_matrix = cdist_dtw(dtrain_day, n_jobs=6)
linkage = complete(distance_matrix)
#Create 6 clusters from the linkage matrix
clusters_ = fcluster(linkage, nb_cluster, criterion='maxclust')
clusters_ = pd.DataFrame(clusters_, index=dtrain_day.index, columns=["cluster"])
clusters = pd.DataFrame(index=dtrain_day.index)
clusters['cluster'] = clusters_
#Aggregate the two smallest clusters to make the clusters more balanced
repartition = [sum(clusters['cluster']==c) for c in pd.unique(clusters['cluster'])]
min1 = np.argmin(repartition)
repartition[min1] = max(repartition)
min2 = np.argmin(repartition)
clusters['cluster'] = np.where(clusters['cluster'].values == min1+1, min2+1, clusters['cluster'].values)
print("CLUSTERS:", np.sort(pd.unique(clusters['cluster'])))

#Classifier to predict new clusters of the data from Dens and Dtest
y_weather = data[['X(t)','temperature(t)']].resample('4H').mean()
X_train = pd.DataFrame(index=dtrain_day.index)
X_train['label'] = clusters['cluster']
#Create features for the classifier
for w in range(h,W+h):
    X_train['cluster-'+str(w)] = X_train['label'].shift(w)
    X_train['Consumption-'+str(w)] = y_weather['X(t)'].shift(w)
X_train['Weather'] = y_weather['temperature(t)']
X_train['hour'] = X_train.index.hour
X_train['Weekday'] = X_train.index.weekday
X_train.dropna(inplace=True)
Y_train = X_train[['label']].astype(int)
X_train.drop(['label'], axis=1, inplace=True)
clusters_col = X_train.columns[[col.startswith('cluster-') for col in X_train.columns]]
num_cols = list(set(X_train.columns[X_train.dtypes == float]) - set(clusters_col))
cat_cols = list(set(X_train.columns) - set(num_cols))
X_train = pd.get_dummies(X_train, columns=cat_cols)
scaler_x = MinMaxScaler(feature_range=(0, 1))
X_train[num_cols] = scaler_x.fit_transform(X_train[num_cols])
Y_train = pd.get_dummies(Y_train, columns = ['label'])
print("X_train", X_train.columns)
#Train Classifier
classifier = mod.classifier(X_train, Y_train, clusters['cluster'])

#Create empty datasets to store the predicted values for each expert
predicted_ens = pd.DataFrame(0, index=dens.index, columns = ["CNN_LSTM", "CNN", "GRU", "LSTM","RNN", "FNN", "LGB", "LR",
                                                          "RF", "XGB", "SVR", "KNN"])
predicted_test = pd.DataFrame(0, index=dtest.index, columns = ["CNN_LSTM", "CNN", "GRU", "LSTM","RNN", "FNN", "LGB", "LR",
                                            "RF", "XGB", "SVR", "KNN", "COP_none", "COP_exp", "COP_inverse", "COP_sum"])

#Create empty datasets to store the predictions of the classifier
predicted_clusters_ens = pd.DataFrame(0, columns=["cluster_" + str(k) for k in np.sort(pd.unique(clusters['cluster']))] , index=dens_day.index)
predicted_clusters_test = pd.DataFrame(0, columns=["cluster_" + str(k) for k in np.sort(pd.unique(clusters['cluster']))], index=dtest_day.index)
experts_models = ["CNN_LSTM", "CNN", "GRU", "LSTM","RNN", "FNN", "LGB", "LR",
                                                          "RF", "XGB", "SVR", "KNN"]

#For each expert, train a model for each cluster
stored_models = {}
for c in np.sort(pd.unique(clusters['cluster'])):
    print(c)
    indexes = [i + timedelta(hours=k) for i in clusters[clusters['cluster'] == c].index for k in range(4)]
    model = tpd.fit_all_models(dtrain.loc[indexes], c)
    stored_models[c] = model

#Cluster forecast for each value from Dens
for index_, day in dens_day.iterrows():
    print("DENS:", str(index_))
    X_ens = pd.DataFrame(0, index=[index_], columns=X_train.columns)
    #Create features
    for w in range(h, W+h):
        X_ens['cluster-' + str(w) + "_" + str(clusters['cluster'].iloc[-w]) + '.0'] = 1
        X_ens['Consumption-' + str(w)] = y_weather['X(t)'].loc[index_ - timedelta(hours=4 * w)]
    X_ens['Weather'] = y_weather['temperature(t)'].loc[index_]
    X_ens['hour_' + str(X_ens.index.hour[0])] = 1
    X_ens['Weekday_' + str(X_ens.index.weekday[0])] = 1
    #Prediction
    pred = classifier.predict(X_ens)[0]
    #Round prediction up to next value with 1 decimal place
    pred = np.round(pred, 1)
    #If the sum of the value is not one, fill the minimum value to complete
    if sum(pred) < 0.9:
        m = np.where(pred > 0, pred, np.inf).argmin()
        pred[m] += 0.1
    predicted_clusters_ens.loc[index_] = pred
    #Find the actual cluster value by computing the distance matrix with DTW
    new_cluster = clusters['cluster'].values[np.argmin(cdist_dtw(dtrain_day, day))]
    clusters = clusters.append(pd.DataFrame(new_cluster, index=[index_], columns=["cluster"]))
    dtrain_day = dtrain_day.append(dens_day.loc[index_])

for c in predicted_clusters_ens.columns:
    print(c, ' ', sum(predicted_clusters_ens[c].values))

#Predict all values from Dens by using the expert models trained on the specific cluster from which the value belongs
for c in stored_models.keys():
    col = "cluster_" + str(c)
    indexes = [i + timedelta(hours=k) for i in predicted_clusters_ens[predicted_clusters_ens[col] > 0].index for k in
               range(4)]
    if dens.loc[indexes].shape[0] > 0:
        for m in stored_models[c]:
            pred = np.array(tpd.prediction(stored_models[c][m], dens.loc[indexes], m, c))
            #Weight the prediction by the weight given by the classifier to the cluster
            predicted_ens[m].loc[indexes] += pred * np.repeat(predicted_clusters_ens[col][predicted_clusters_ens[col] > 0].values, 4, axis=0)
print("PREDICTED ENS", predicted_ens.head(10))

# Train the Convex Optimisation Problems on each cluster
stored_models_cop = {}
for c in stored_models.keys():
    indexes = [i + timedelta(hours=k) for i in clusters['2009'][clusters['cluster'] == c].index for k in range(4)]
    if dens.loc[indexes].shape[0]>0:
        cop_models = tpd.fit_all_COP(dens.loc[indexes], predicted_ens.loc[indexes])
        stored_models_cop[c] = cop_models
    else:
        print('No representative')
        stored_models_cop[c] = stored_models_cop[c-1]

# Predict clusters for Dtest
for index_, day in dtest_day.iterrows():
    print("DTEST:", str(index_))
    X_test = pd.DataFrame(0, index=[index_], columns=X_train.columns)
    for w in range(h, W+h):
        X_test['cluster-' + str(w) + "_" + str(clusters['cluster'].iloc[-w]) + '.0'] = 1
        X_test['Consumption-' + str(w)] = y_weather['X(t)'].loc[index_ - timedelta(hours=4 * w)]
    X_test['Weather'] = y_weather['temperature(t)'].loc[index_]
    X_test['hour_'+str(X_test.index.hour[0])] = 1
    X_test['Weekday_'+str(X_test.index.weekday[0])] = 1
    pred = classifier.predict(X_test)[0]
    pred = np.round(pred, 1)
    if sum(pred) < 0.9:
        m = np.where(pred > 0, pred, np.inf).argmin()
        pred[m] += 0.1
    predicted_clusters_test.loc[index_] = pred
    new_cluster = clusters['cluster'].values[np.argmin(cdist_dtw(dtrain_day, day))]
    clusters = clusters.append(pd.DataFrame(new_cluster, index=[index_], columns=["cluster"]))
    dtrain_day = dtrain_day.append(dtest_day.loc[index_])

# Forecast values of Dtest according to the clusters, for all experts and the Convex Optimization Problem
for c in stored_models.keys():
    col = "cluster_" + str(c)
    indexes = [i + timedelta(hours=k) for i in predicted_clusters_test[predicted_clusters_test[col] >0].index for k in
               range(4)]
    if dtest.loc[indexes].shape[0] > 0:
        for m in stored_models[c]:
            pred = np.array(tpd.prediction(stored_models[c][m], dtest.loc[indexes], m, c))
            predicted_test[m].loc[indexes] += pred * np.repeat(predicted_clusters_test[col][predicted_clusters_test[col] > 0].values, 4, axis=0)
        for m in stored_models_cop[c]:
            pred = np.array(tpd.prediction(stored_models_cop[c][m], predicted_test[predicted_test.columns[:-4]].loc[indexes], m, c))
            predicted_test[m].loc[indexes] += pred * np.repeat(predicted_clusters_test[col][predicted_clusters_test[col] > 0].values, 4, axis=0)
    else:
        print("NO Corresponding values for cluster ", c)

predicted_clusters_ens['cluster'] = np.argmax(predicted_clusters_ens.values, axis=1)
predicted_clusters_test['cluster'] = np.argmax(predicted_clusters_test.values, axis=1)
predicted_ens = pd.merge(predicted_ens, global_predicted_ens, left_index=True, right_index=True)
predicted_test = pd.merge(predicted_test, global_predicted_test, left_index=True, right_index=True)

# Compute RMSE and NRMSE for each model for each cluster
predicted_clusters_ens = predicted_clusters_ens[['cluster']]
rmse_models = pd.DataFrame(columns=predicted_ens.columns, index=np.sort(pd.unique(predicted_clusters_ens['cluster'])))
nrmse_models = pd.DataFrame(columns=predicted_ens.columns, index=np.sort(pd.unique(predicted_clusters_ens['cluster'])))
for model in rmse_models.columns:
    for c in rmse_models.index:
        indexes = [i + timedelta(hours=k) for i in predicted_clusters_ens[predicted_clusters_ens['cluster']==c].index for
                   k in range(4)]
        rmse_models[model][c] = mod.rmse(dens.loc[indexes]['X(t)'],
                                         predicted_ens.loc[indexes][model])
        nrmse_models[model][c] = mod.nrmse(dens.loc[indexes]['X(t)'],
                                        predicted_ens.loc[indexes][model])

# Scale rmse and nrmse to make them comparable
rmse_models = (rmse_models-rmse_models.min().min())/(rmse_models.max().max()-rmse_models.min().min())
nrmse_models = (nrmse_models-nrmse_models.min().min())/(nrmse_models.max().max()-nrmse_models.min().min())
error = rmse_models + nrmse_models

# Select and combine the best models for each cluster
best_pred_ens = pd.DataFrame(index=predicted_ens.index, columns = ['best_' + str(k) for k in range(nb_experts)])
best_pred_test = pd.DataFrame(index=predicted_test.index, columns = ['best_' + str(k) for k in range(nb_experts)])
for c in range(len(error.index)):
    cl=error.index[c]
    indexes_ens = [i + timedelta(hours=k) for i in predicted_clusters_ens[predicted_clusters_ens['cluster'] == c].index for
               k in range(4)]
    indexes_test = [i + timedelta(hours=k) for i in predicted_clusters_test[predicted_clusters_test['cluster'] == c].index
                   for k in range(4)]
    best_models = error.columns[np.argsort(error.iloc[c]).values[:nb_experts]]
    best_pred_ens.loc[indexes_ens] = predicted_ens[best_models].loc[indexes_ens]
    best_pred_test.loc[indexes_test] = predicted_test[best_models].loc[indexes_test]

### MOE
dens[num_cols_d] = dens[num_cols_d].astype('float32')
best_pred_ens = best_pred_ens.astype('float32')
m = mod.fit_MOE(dens.drop(['X(t)'], axis=1), best_pred_ens, dens[['X(t)']])
if best_pred_test.shape[0] < dtest.shape[0]:
    dtest_ = dtest.iloc[:best_pred_test.shape[0]]
else :
    dtest_ = dtest
dtest_[num_cols_d] = dtest_[num_cols_d].astype('float32')
best_pred_test = best_pred_test.astype('float32')

X_test_array = dtest_.drop(['X(t)'], axis=1).values.reshape((dtest_.drop(['X(t)'], axis=1).shape[0], dtest_.drop(['X(t)'], axis=1).shape[1], 1))
y_hat_test_array = best_pred_test.values.reshape((best_pred_test.shape[0], best_pred_test.shape[1]))
MOE_prediction = m.predict([X_test_array, y_hat_test_array])
MOE_prediction = MOE_prediction.reshape(-1, 1)
MOE_prediction = pd.DataFrame(MOE_prediction, index=dtest_.index)

best_pred_test['AVG'] = np.mean(best_pred_test, axis=1)
best_pred_test['MOE'] = MOE_prediction

best_pred_test = best_pred_test * (max_y - min_y) + min_y
best_pred_test[best_pred_test < 0] = 0
best_pred_test[best_pred_test > max_y] = max_y
predicted_test.fillna(method='ffill', inplace=True)
predicted_test = predicted_test.astype('float32')
predicted_test = predicted_test * (max_y - min_y) + min_y
predicted_test[predicted_test < 0] = 0
predicted_test[predicted_test > max_y] = max_y

# Save Data
prediction = pd.concat([best_pred_test, predicted_test], axis=1)
prediction.to_pickle(
os.path.join(repos_path, "results", "PSF_DTW", str(str(h) + "_" + str(nb_experts) + "_" + str(nb_cluster) + "_prediction_correct_argmax.pkl")))

error.to_pickle(os.path.join(repos_path, "results", "PSF_DTW", str(str(h) + "_" + str(nb_experts) + "_" + str(nb_cluster) + "_error_correct_argmax.pkl")))
