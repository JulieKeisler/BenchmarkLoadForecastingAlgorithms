from datetime import datetime
import pandas as pd
import holidays
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# define features for individual learners
def features(df, H, name):
    data=df.copy()
    data['Weekday'] = data.index.weekday
    data['Weekend'] = data['Weekday'] > 4
    data['Month'] = data.index.month

    fr_holidays = []
    for date, name_ in sorted(holidays.FR(years=[2006, 2007, 2008, 2009, 2010, 2011]).items()):
        fr_holidays.append(date)
    date1 = data.index.date[0]
    date2 = data.index.date[-1]

    def filter_func(date):
        return date >= date1 and date <= date2

    fr_holidays = list(filter(filter_func, fr_holidays))
    data['Holidays'] = np.nan
    for date in data.index.date:
        data['Holidays'][str(date)] = date in fr_holidays
    data = data.resample('D').mean()
    data.drop(['X(t)'], axis=1)
    data['Holidays'] = data['Holidays'].astype(bool)
    data['Date'] = data.index.day
    data['Year'] = data.index.year
    data['Cluster-h'] = data['cluster_' + name].shift(H)
    data.dropna(inplace=True)
    return(data)

# fit classifier to predict clusters on ensemble and test datasets
def fit_classifier(data, name):
    estimators = [('xgb', XGBClassifier(n_estimators=100, booster='gbtree', eta=0.01, gamma=10, max_depth=5)),
                  ('tree', DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=10, splitter='best')),
                  ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, bootstrap=False)),
                  ('knn', KNeighborsClassifier(n_neighbors=10, weights='distance', leaf_size=2))
                  ]
    clf = StackingClassifier(estimators=estimators,
                             final_estimator=SVC(C=2))
    clf.fit(data.drop(['cluster_' + name], axis=1), data['cluster_' + name])
    return(clf)

# train classifier on dtrain and predict clusters of dens and dtest
def train_pred_classifier(data, H, name):
    """
    Parameters
    ----------
    data: dataset containing dtrain, dens, dtest
    H: forecast horizon
    name: type of clustering algorithm
    """
    H = max(H//24,1)
    # For hourly clustering no need for a classifier
    if name == 'hourly' :
        return(data['2006':'2008'], data['2009':'2010'])
    else :
        df = features(data, H, name)
        dftrain = df['2006':'2008']
        dftest = df['2009':'2010']
        clf = fit_classifier(dftrain, name)
        prediction = pd.DataFrame(index=dftest.index)
        prediction['cluster'] = clf.predict(dftest.drop(['cluster_'+name], axis=1))
        Dtrain = data['2006':'2008']
        Dtest = data['2009':'2010']
        for date in Dtest.index.date:
            if date in prediction.index.date:
                date1 = datetime.strftime(date, "%Y-%m-%d")
                Dtest['cluster_' + name][date1] = prediction['cluster'][date1]
        return (Dtrain, Dtest)
