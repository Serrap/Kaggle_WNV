'''
Paolo Serra, 10 / 03 / 2017
Program to estimate the WNV presence for Kaggle competition, see:
https://www.kaggle.com/c/predict-west-nile-virus
'''

import pdb
# general data manipulation and plotting
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.neighbors import KernelDensity
from pylab import *
import ast

# Dimensionality reduction
from sklearn.decomposition import PCA

# Data pre-processing
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, make_scorer
from imblearn.over_sampling import SMOTE
from datetime import datetime
import time
from sklearn.utils import shuffle
import datetime as dt

# model metrics
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ML algorithms
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier


# ******* Data exploration / visualization routines ****
def plot_class_imbalance(df):
    # Function to plot and output the class imbalance
    plt.rcParams['figure.facecolor'] = 'white'
    sns.countplot('WnvPresent', data = df)
    perc_WnvAbsent = df['WnvPresent'].value_counts(normalize = True)[0]
    perc_WnvPresent = df['WnvPresent'].value_counts(normalize = True)[1]
    print('There is ', perc_WnvPresent, ' %', 'of Presence of Wnv')
    plt.show()

def plot_counts_columns(df, label):
    # Function to compute and plot the relative counts of 
    # various classes for a given feature
    count = df[label].value_counts(normalize = True)
    names = df[label].unique()
    indices = np.argsort(count)[::-1]
    count, names = zip(*sorted(zip(count, names)))
    plt.barh(range(len(names)), count, align = 'center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Percentages', fontsize=17)
    plt.ylabel('Features', fontsize=17)
    plt.title('Percentage of each class', fontsize=17)
    plt.show()

def plot_corr_matrix(df):
    # Function to plot correlation matrix
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    print('Correlation matrix')
    sns.heatmap(corrmat, annot = True, square=True)
    plt.show()

def plot_grouped_features(df, feature, target):
    # Function to make a dictionary for the relative
    # frequency of the Nile virus for each categoty
    feature_dict = dict()
    for group, frame in df.groupby([feature]):
        avg = np.average(frame[target])
        feature_dict[group] = avg
    return feature_dict

def plot_pca(df, target):
    # Function to visualize our dataset with PCA--
    # basically a projection onto the hyperplane that maximizes variance
    del df['WnvPresent']
    del df['Trap']
    del df['Species']
    pca = PCA()
    x_pca = pca.fit_transform(df)
    colors = ['red' if i==1 else 'blue' for i in y_train]
    plt.scatter(x_pca[:,0], x_pca[:,1], s=30, alpha=.5, c=colors)
    plt.xlabel('PCA-1', fontsize=16)
    plt.ylabel('PCA-2', fontsize=16)
    plt.show()

def map_dict_dataframe(df, dict, label):
    avg_label = 'avg_' + label
    df[avg_label] = df[label].map(dict)
    df[avg_label] = df[avg_label].fillna(df[avg_label])
    return df 

def make_heatmap(df, mapdata, year, month):
    # Function to make heatmaps for various features
    data = df[['Date', 'Trap', 'Longitude', 'Latitude', 'WnvPresent', 'year', 'month']]
    data = data[(data.year == year) & (data.month == month)]
    alpha_cm = plt.cm.Reds
    alpha_cm._init()
    alpha_cm._lut[:-3,-1] = abs(np.logspace(0, 1, alpha_cm.N) / 10 - 1)[::-1]
    aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
    lon_lat_box = (-88, -87.5, 41.6, 42.1)
    sigthings = data[data['WnvPresent'] > 0]
    sigthings = sigthings.groupby(['Date', 'Trap','Longitude', 'Latitude']).max()['WnvPresent'].reset_index()
    X = sigthings[['Longitude', 'Latitude']].values
    xv,yv = np.meshgrid(np.linspace(-88, -87.5, 100), np.linspace(41.6, 42.1, 100))
    gridpoints = np.array([xv.ravel(),yv.ravel()]).T
    if X.shape[0] > 0:
        kd = KernelDensity(bandwidth=0.02)
        kd.fit(X)
        zv = np.exp(kd.score_samples(gridpoints).reshape(100,100))
    else:
        zv = gridpoints[:, 0].reshape(100,100) * 0.0
    plt.figure(figsize=(10,14))
    plt.imshow(mapdata, 
               cmap=plt.get_cmap('gray'), 
               extent=lon_lat_box, aspect=aspect)
    plt.imshow(zv, 
           origin='lower', 
           cmap=alpha_cm, 
           extent=lon_lat_box, 
           aspect=aspect)
    locations = data[['Longitude', 'Latitude']].drop_duplicates().values
    plt.scatter(locations[:,0], locations[:,1], marker='x')
    plt.savefig('heatmap_' + str(year) + '_' + str(month) + '.png')
    plt.show()
    
def estimate_avg_mosquitos(df):
    # Function to try and estimate number of mosquitos
    # from number of rows
    temp_count = df.groupby(['Date', 'Address', 'Trap']).count()
    count = temp_count['Latitude'].reset_index().rename(columns={'Latitude': 'row_count'})
    test = pd.merge(df, count, on = ['Date', 'Address', 'Trap']) 
    final_data = test #pd.merge(test1, test2, on = ['row_count'])
    return final_data


def create_dates(df):
    # Function to create dates for train/test datasets
    df['year'] = df['month'] = df['week'] = df['past_week'] = df['day'] = 0
    num = df.shape[0]
    for i in range(num):
        year = int(df['Date'].values[i].split('-')[0])
        month = int(df['Date'].values[i].split('-')[1])
        day = int(df['Date'].values[i].split('-')[2])
        df['year'].values[i] = year
        df['month'].values[i] = month
        df['day'].values[i] = day
        df['week'].values[i] = float(dt.date(year, month, day).strftime("%V"))
        df['past_week'].values[i] =  df['week'].values[i] - 1
    return df  

def delete_columns(df):
    del df['Street']
    del df['AddressNumberAndStreet']
    del df['AddressAccuracy']
    return df

def plot_tseries_trap(df, label):
    # Function to plot time series for various quantities
    tseries = df[df.Trap == label].groupby('Date').agg({'WnvPresent': np.sum})
    tseries.plot()
    plt.show()
    
def new_features_proba(df, label):
# Function to assess probability associated to traps, block, species etc...
#tot_WnvPresent = df.WnvPresent.sum()  
    groupby_label = df.groupby([label]).agg({'WnvPresent': np.average}).rename(columns={'WnvPresent': 'avg_'+label}) #/ tot_WnvPresent
    groupby_label = groupby_label.reset_index()
    return groupby_label

def merge_features(df, labels):
    # Function to merge various features
    for lb in labels:
        df_label = new_features_proba(df, lb)
        df = pd.merge(df, df_label, on = lb)
    return df

def past_week_weather(df):
    # Function to compute lagged variables
    output_data = pd.DataFrame()
    dates = df.year.unique()
    for year in dates:
        data_year = df[df['year'] == year]
        data_Wnv = data_year.groupby(['year', 'month', 'past_week']).agg({'Depart': np.sum, 'PrecipTotal': np.sum, 'Tmax': np.sum, \
        'DewPoint': np.sum, 'WetBulb': np.sum, 'StnPressure': np.sum, \
        'Tmin': np.sum})
        output_data = output_data.append(data_Wnv)
    output_data = output_data.rename(columns = lambda x: 'past_' + x)
    output_data = output_data.reset_index(level=['year', 'month', 'past_week'])
    output_data['week'] = output_data['past_week'] + 1
    return output_data
    
def poly_weather_features(df):
    # Function to make polynomial features
    weather_combo = df[['past_Depart', 'past_DewPoint', 'past_WetBulb', 'past_Tmin', 'past_Tmax', 'past_PrecipTotal', 'past_StnPressure']]
    #poly = PolynomialFeatures(2)
    #weather_combo = pd.DataFrame(poly.fit_transform(weather_combo))
    weather_combo['year'] = df['year']
    weather_combo['month'] = df['month']
    weather_combo['past_week'] = df['past_week']
    weather_combo['week'] = df['week']
    return weather_combo
    

# Random Forest Classifier
def ml_rf_class(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier()
    grid = GridSearchCV(rf, {'max_depth': range(2, 10),
                             'n_estimators': [10, 20, 50, 100],
                             'max_features': [1, 3, 5, X_train.shape[1]],
        }, cv = 5, n_jobs=-1)
    gridfit = grid.fit(X_train, y_train)
    rf = gridfit.best_estimator_
    rf_predict_proba = rf.predict_proba(X_test)
    rf_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, rf_predict, average = 'weighted')
    print("Precision, Recall, Fscore, support for Random Forest: ", precision, recall, fscore, support)
    plot_roc_curve(rf_predict_proba, y_test)
    conf_matrix(y_test, rf_predict)
    plot_feature_importance(rf.feature_importances_, rf.estimators_)
    return rf_predict, rf_predict_proba, rf


def ml_linear_ridge(X_train, y_train, X_test, y_test):
    # Linear regression (Ridge)
    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    reg.fit(X_train, y_train)
    reg_predict = reg.predict(X_test)
    reg.score(X_test, y_test, sample_weight=None)
    print(reg.coef_)
    
def ml_forest_regr(X_train, y_train, X_test, y_test):
    # Random Forest Regression
    rf = RandomForestRegressor()
    grid = GridSearchCV(rf, {'max_depth': range(2, 10),
                             'n_estimators': [10, 20, 50, 100],
                             'max_features': [1, 3, 5, X_train.shape[1]],
        }, cv = 5, n_jobs=-1)
    
    gridfit = grid.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    rf = gridfit.best_estimator_
    rf_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    print('Score is: ', rf_score)
    plot_feature_importance(rf.feature_importances_, rf.estimators_)

# KNN classifier 
def ml_knn_class(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler()
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors':[3, 5, 10, 20, 30, 50, 100], "weights": ["uniform", "distance"]}
    grid = GridSearchCV(knn, parameters, cv = 5)
    gridfit = grid.fit(X_train, y_train)
    knn = gridfit.best_estimator_
    knn_predict_proba =  knn.predict_proba(X_test)
    knn_predict = knn.predict(X_test)
    knn_score = knn.score(X_test, y_test)
    print('knn_score: ', knn_score)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, knn_predict, average = 'weighted')
    print("Precision, Recall, Fscore, support for KNN: ", precision, recall, fscore, support)
    plot_roc_curve(knn_predict_proba, y_test)
    conf_matrix(y_test, knn_predict)
    return knn_predict, knn_predict_proba, knn

def ml_adaboost_class(X_train, y_train, X_test, y_test):
    param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]}
    DTC = tree.DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced", max_depth = None)
    ABC = AdaBoostClassifier(base_estimator = DTC)
    # run grid search
    grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, cv = 5, scoring = 'roc_auc')
    gridfit = grid_search_ABC.fit(X_train, y_train)
    ada = gridfit.best_estimator_
    ada_predict_proba = ada.predict_proba(X_test)
    ada_predict = ada.predict(X_test)
    ada_score = ada.score(X_test, y_test)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, ada_predict, average = 'weighted')
    print("Precision, Recall, Fscore, support for AdaBoost: ", precision, recall, fscore, support)
    plot_roc_curve(ada_predict_proba, y_test)
    conf_matrix(y_test, ada_predict)
    return ada_predict, ada_predict_proba, ada

# compute and plot ROC curve
def plot_roc_curve(predictions, target):
    # Function to compute and plot the ROC curve
    fpr, tpr, threshold = metrics.roc_curve(target, predictions[:,1], pos_label = True)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver operating characteristic curve', fontsize=17)
    plt.plot(fpr, tpr, 'blue', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=17)
    plt.xlabel('False Positive Rate',  fontsize=17)
    plt.show()

def plot_feature_importance(importances, estimators):
    # Function to compute and plot the feature importance 
    std = np.std([tree.feature_importances_ for tree in estimators],axis=0)
    indices = np.argsort(importances)[::-1]
    #print("Feature ranking:")
    #for f in range(df.shape[1]):
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    names = df.columns
    del df['row_count']
    del df['row_count**2 * month']
    names = df.columns
    #pdb.set_trace()
    importances, names = zip(*sorted(zip(importances, names)))
    plt.barh(range(len(names)), importances, align = 'center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Percentages', fontsize=17)
    plt.ylabel('Features', fontsize=17)
    plt.title('Importance of each feature', fontsize=17)
    plt.show()
    
def conf_matrix(y_test, predict):
    # Function to compute and plot the confusion matrix
    cm = confusion_matrix(y_test, predict)
    #cm = cm_good.astype('float') / cm_good.sum(axis=1)[:, np.newaxis]
    print('Confusion matrix')
    print(cm)
    sns.heatmap(cm, annot = True, square=True)
    plt.show()

# Load all datasets
df_train = pd.read_csv("./train.csv")
df_train_copy = df_train.copy()
df_train = df_train.convert_objects(convert_numeric=True)
df_train = df_train.dropna()
target = df_train['WnvPresent']

df_weather = pd.read_csv("./weather.csv")
df_weather = df_weather.convert_objects(convert_numeric = True)
df_weather = df_weather.fillna(method='ffill')

output = pd.read_csv('./sampleSubmission.csv')
mapdata = np.loadtxt("./mapdata_copyright_openstreetmap_contributors.txt")

# test set from Kaggle 
df_test_kaggle = pd.read_csv("./test.csv")
df_test_kaggle_copy = df_test_kaggle.copy()

# make a test-set to test the code
#df_train, df_test = train_test_split(df_train_copy, test_size=0.2)
#y_target = df_test['WnvPresent']
df_test = df_test_kaggle

labels = ['Trap', 'Species', 'Block']

# Data exploration
df_train.describe()
plot_class_imbalance(df_train)
plot_counts_columns(df_train, 'Species')
plot_corr_matrix(df_train)

df_train = create_dates(df_train)
# make heatmaps for various years
#for year in df_train.year.unique():
#    for month in df_train.month.unique():
#        make_heatmap(df_train, mapdata, year, month)

df_train = delete_columns(df_train)
df_train = pd.merge(df_train, df_weather[df_weather['Station'] == 1], on = 'Date')

df_test = create_dates(df_test)
df_test = delete_columns(df_test)
df_test = pd.merge(df_test, df_weather[df_weather['Station'] == 1], on = 'Date')

df_train = merge_features(df_train, labels)
dict_Trap = pd.Series(df_train.avg_Trap.values,index=df_train.Trap).to_dict()
dict_Species = pd.Series(df_train.avg_Species.values,index=df_train.Species).to_dict()
dict_Block = pd.Series(df_train.avg_Block.values,index=df_train.Block).to_dict()


df_test['avg_Trap'] = df_test['Trap'].map(dict_Trap)
df_test['avg_Trap'] = df_test['avg_Trap'].fillna(df_test['avg_Trap'].mean())
df_test['avg_Species'] = df_test['Species'].map(dict_Species)
df_test['avg_Species'] = df_test['avg_Species'].fillna(df_test['avg_Species'].mean())
df_test['avg_Block'] = df_test['Block'].map(dict_Block)
df_test['avg_Block'] = df_test['avg_Block'].fillna(df_test['avg_Block'].mean())

df_train = estimate_avg_mosquitos(df_train)
df_test = estimate_avg_mosquitos(df_test)

past_weather_train = past_week_weather(df_train)
past_weather_test = past_week_weather(df_test)


df_train = df_train[['Latitude', 'Longitude', 'Sunrise', \
                     'Sunset', 'Tmax', 'DewPoint', \
                     'WnvPresent', 'year', 'month', 'week', 'past_week','avg_Trap', 'avg_Species', 'row_count']]



                     #df_train = pd.get_dummies(df_train)

df_test = df_test[['Latitude', 'Longitude', \
            'Sunrise', 'Sunset', 'Tmax', 'DewPoint', \
            'year', 'month', 'week', 'past_week',  'avg_Trap', 'avg_Species', 'row_count']]

df_test = pd.get_dummies(df_test)
 
weather_feat_train = poly_weather_features(past_weather_train)
weather_feat_test = poly_weather_features(past_weather_test)
df_train = pd.merge(df_train, weather_feat_train, on = ['year', 'month', 'week', 'past_week'])
df_test = pd.merge(df_test, weather_feat_test, on = ['year', 'month', 'week', 'past_week'])

# modify the month feature
df_train.month[df_train.month == 9] = 7
df_train.month[df_train.month == 10] = 6
df_test.month[df_test.month == 9] = 7
df_test.month[df_test.month == 10] = 6


df_train['row_count**2 * month'] = df_train.row_count**2 * df_train.month
df_test['row_count**2 * month'] = df_test.row_count**2 * df_test.month

target = df_train['WnvPresent']
del df_train['WnvPresent']
del df_train['year']
#del df_train['month']
del df_train['week']
del df_train['past_week']

del df_test['year']
#del df_test['month']
del df_test['week']
del df_test['past_week']

df = df_train
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = 0.2, random_state = 0, stratify = target)

# Oversample the minority class
sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
X_train = X_train_res
y_train = y_train_res

# Visualize our dataset with PCA--
# basically a projection onto the hyperplane that maximizes variance
#pca = PCA()
#x_pca = pca.fit_transform(X_train)
#colors = ['red' if i==1 else 'blue' for i in y_train]
#plt.scatter(x_pca[:,0], x_pca[:,1], s=30, alpha=.5, c=colors)
#plt.xlabel('PCA-1', fontsize=16)
#plt.ylabel('PCA-2', fontsize=16)
#plt.show()


# Check multiple algorithms
predict_ada, predict_proba_ada, ada_est = ml_adaboost_class(X_train, y_train, X_test, y_test)
predict_knn, predict_proba_knn, knn_est = ml_knn_class(X_train, y_train, X_test, y_test)
predict_rf, predict_proba_rf, rf_est = ml_rf_class(X_train, y_train, X_test, y_test)

predict_proba_ensemble = (predict_proba_ada + predict_proba_knn + predict_proba_rf) / 3.

# output data for Kaggle prediction
#output_proba_pred = (ada_est.predict_proba(df_test) + knn_est.predict_proba(df_test) + rf_est.predict_proba(df_test)) / 3.
#output['WnvPresent'] = output_proba_pred[:, 1]
#output.to_csv('Nile.csv', index=False)
#print('The ratio of WNV virus in the test set is: ', output[output.WnvPresent > 0.5].shape[0] /output.shape[0])
print('Ensemble ROC')               
plot_roc_curve(predict_proba_ensemble, y_test)

