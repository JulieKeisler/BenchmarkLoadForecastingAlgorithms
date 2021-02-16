# import
from datetime import datetime
import SimpSOM as sps
from sklearn.cluster import KMeans
import dtwsom
from pyclustering.nnet.som import type_conn
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

# Reshape data to subsequences that will be clustered
def pivot(data):
    df = data.copy()
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df = df.pivot(index='date', columns='hour')
    df.dropna(inplace=True)
    return(df)

# Return labels computing with SOM function using DTW distance
def som_dtw(data):
    """
    :param data: DataFrame with data to cluster, one line corresponds to one day
    :return: Dataframe with one label per day
    """
    rows = 2
    cols = 3
    structure = type_conn.grid_eight
    net = dtwsom.DtwSom(rows, cols, structure)
    net.train(list(data.values), 30)
    # plot some information about the network
    net.save_distance_matrix("Distance_Matrix.png")
    net.save_winner_matrix("Winner_Matrix.png")
    n_neurons = net._size
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for neuron_index in range(n_neurons):
        col = math.floor(neuron_index / 2)
        row = neuron_index % 2
        neuron_weights = net._weights[neuron_index]
        axs[row, col].plot(np.arange(len(neuron_weights)), neuron_weights, label=str(neuron_index))
        axs[row, col].set_ylabel("Neuron: " + str(neuron_index))
    fig.savefig("Neurons.png")
    labels = pd.DataFrame(index=data.index)
    labels['cluster']=np.nan
    c = 0 # cluster label
    for cluster in net.capture_objects:
        for id in cluster:
            labels['cluster'].iloc[id] = c
        c += 1
    return labels

# SOM with euclidean distance
def som_euclidean(data, method):
    """
    :param data: DataFrame with data to cluster, one line corresponds to one day
    :return: Dataframe with one label per day
    """
    labels = pd.DataFrame(index=data.index)
    rows = 20
    cols = 20
    net = sps.somNet(rows, cols, data.values, PBC=True)
    net.train(0.01, 20000)
    prj = np.array(net.project(data.values))
    if method=='euclidan_24':
        kmeans = KMeans(n_clusters=24, random_state=0).fit(prj)
    else:
        kmeans = KMeans(n_clusters=4, random_state=0).fit(prj)
    print("KMEANS:", kmeans.labels_)
    labels['cluster'] = kmeans.labels_
    print(labels.head())
    return(labels)


def resample_data(data, labels):
    """
    :param data: DataFrame that has been clustered
    :param labels: Labels corresponding to data
    :return: DataFrame with a new column "cluster", with a cluster label for each hour
    """
    data["cluster"] = np.nan
    for date in labels.index.date:
        if date in data.index.date:
            date1 = datetime.strftime(date, "%Y-%m-%d")
            label = labels['cluster'].loc[date1]
            data['cluster'][date1] = label
    data.dropna(inplace=True)
    return(data)

# Hourly clustering
def clustering_hourly(df):
    df['cluster'] = df.index.hour
    return(df)

# define clustering type
def clustering_type(method, data):
    if method == 'hourly' :
        return(clustering_hourly(data))
    elif method == 'dtw_4' or method == 'dtw_24':
        data.rename(columns={'clusters': 'cluster'}, inplace=True)
        return(data)
    elif method == 'euclidean_24' or method=='euclidean_4' :
        df_pivot = pivot(data)
        if method == 'dtw' :
            return(resample_data(data, som_dtw(df_pivot), "dtw"))
        else :
            return(resample_data(data, som_euclidean(df_pivot, method)))





