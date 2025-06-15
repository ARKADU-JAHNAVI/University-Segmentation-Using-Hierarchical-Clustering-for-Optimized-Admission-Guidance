# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:08:53 2025

@author: arkad
"""
pip install sweetviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sweetviz
from AutoClean import AutoClean

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics
from clusteval import clusteval

from sqlalchemy import create_engine, text
from urllib.parse import quote

uni = pd.read_excel(r"C:\Users\arkad\Downloads\Data Sets\University_Clustering.xlsx")

uni.shape
uni.describe
uni.info()
uni.columns




user = 'root'  # user name
pw = 'JANU1604'  # password
db = 'univ_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


uni.to_sql('univ_tbl', index = False, chunksize = 1000, if_exists = 'replace', con = engine)

# EDA
uni.duplicated()
uni.isna().sum()

# Replacing Missing Values
# Median Imputator
from sklearn.impute import SimpleImputer
med_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
uni['SAT'] = pd.DataFrame(med_imputer.fit_transform(uni[['SAT']]))
uni['SAT'].isna().sum()

# med_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
uni['GradRate'] = pd.DataFrame(med_imputer.fit_transform(uni[['GradRate']]))
uni['GradRate'].isna().sum()

uni['SFRatio'] = pd.DataFrame(med_imputer.fit_transform(uni[['SFRatio']]))
uni['SFRatio'].isna().sum()


import seaborn as sns
sns.boxplot(uni['SAT'])
sns.boxplot(uni['Top10'])
sns.boxplot(uni['Accept'])
sns.boxplot(uni['SFRatio'])
sns.boxplot(uni['Expenses'])
sns.boxplot(uni['GradRate'])


df = uni.drop(['UnivID', 'Univ', 'State'], axis = 1)
df.columns


num_cols = df.select_dtypes(include = [np.number]).columns
for col in num_cols:
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower = df[col].quantile(0.25) - (1.5 * iqr)
    upper = df[col].quantile(0.75) + (1.5 * iqr)
    
    outliers = np.where(df[col] > upper,True, np.where(df[col] < lower, True, False))
    print(f"Outliers in {col}: {outliers.sum()}")



# iqr = df['SAT'].quantile(0.75) - df['SAT'].quantile(0.25)
# lower = df['SAT'].quantile(0.25) - (1.5 * iqr)
# upper = df['SAT'].quantile(0.75) + (1.5 * iqr)

# outliers_inv = np.where(df.SAT > upper, True,np.where(df.SAT < lower, True, False))


# pip install feature_engine

from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr',
                        tail = 'both',
                        fold = 1.5,
                        variables = ["SAT", "Top10", "Accept", "SFRatio", "Expenses", "GradRate"])

df_1 = winsor_iqr.fit_transform(df[['SAT', "Top10","Accept", "SFRatio", "Expenses", "GradRate"]])
df_1
sns.boxplot(df_1['SAT'])
sns.boxplot(df_1['Top10'])
sns.boxplot(df_1['Accept'])
sns.boxplot(df_1['SFRatio'])

num = df_1.select_dtypes(include = [np.number]).columns
for i in num:
    iqr = df_1[i].quantile(0.75) - df_1[i].quantile(0.25)
    lower = df_1[i].quantile(0.25) - (1.5 * iqr)
    upper = df_1[i].quantile(0.75) + (1.5*iqr)
    
    outliers_updated = np.where(df_1[i] > upper, True, np.where(df_1[i] < lower, True, False))
    print(f"outliers in {i}: {outliers_updated.sum()}")
    



#################################################
# ZERO VARIANCE
df_1.dtypes
numeric = df_1.select_dtypes(include = np.number)
numeric.var()
numeric.var() == 0


################################################
# NORMAL Q-Q PLOT
import scipy.stats as stats
import pylab

stats.probplot(df_1.SAT, dist ="norm", plot = pylab)
stats.probplot(df_1.Top10, dist = "norm", plot = pylab)
stats.probplot(df_1.Accept, dist = "norm", plot = pylab)
stats.probplot(df_1.SFRatio, dist ="norm", plot = pylab)
stats.probplot(df_1.Expenses, dist ="norm", plot = pylab)
stats.probplot(df_1.GradRate, dist ="norm", plot = pylab)

# NORMALLY DISTRIBUTED


####################################################
# FEATURE SCALING
# Normalization - MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
df_n = minmax.fit_transform(df_1)
df_final = pd.DataFrame(df_n)
df_final.describe()
df_final= df_final.rename(columns = {0: 'SAT', 1: 'Top10', 2: 'Accept', 3: 'SFRatio', 4: 'Expenses', 5:'GradRate'})

####################################################
# PUSH TO SQL
user = 'root'  # user name
pw = 'JANU1604'  # password
db = 'univ_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

df_final.to_sql("afterPreprcessing", con = engine, if_exists = 'replace', chunksize = 1000,index = False )



#####################################################
# MODEL BUILDING
# Hierarchial Clustering

plt.figure(1, figsize = (16,8))
tree_plot = dendrogram(linkage(df_final, method = "complete"))

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Euclidian Distance")
plt.show()


##################################################
# Applying Agglomerative Clustering
hc1 = AgglomerativeClustering(n_clusters = 2, metric = "euclidean", linkage = "complete")
y_hc1 = hc1.fit_predict(df_final)
y_hc1

hc1.labels_
cluster_labels = pd.Series(hc1.labels_)
df_clust = pd.concat([cluster_labels, df_final], axis = 1)

df_clust = df_clust.rename(columns = {0: "cluster"})
df_clust.head()


################################################
# Cluster Evaluation

metrics.silhouette_score(df_final,cluster_labels)


metrics.davies_bouldin_score(df_final, cluster_labels)

# NOT GOOD CLUSTER

################################################
param_grid = { 
    'n_clusters' : [2,3,4],
    'metric' : ['euclidean', 'manhattan', 'cosine'],
    'linkage' :['complete', 'single', 'average', 'ward']
    }

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

def custom_scorer(estimator, X):
    lables = estimator.fit_predict(X)
    if len(set(lables)) == 1:
        return -1
    return silhouette_score(X,labels)

agg_cluster = AgglomerativeClustering()

gridsearch = GridSearchCV(
    estimator = agg_cluster,
    scoring = custom_scorer,
    param_grid = param_grid,
    cv = 3)

gridsearch.fit(df_final)

print("best parameters: ", gridsearch.best_params_)
print("best score:", gridsearch.best_score_)

best_model = gridsearch.best_estimator_
labels = best_model.fit_predict(df_final)

df_final['cluster'] = labels
print(df_final)
