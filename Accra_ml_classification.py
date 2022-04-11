#!/usr/bin/env python
# coding: utf-8

#%%

# import libraries
import geowombat as gw
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from geowombat.ml import fit
from geowombat.ml import fit_predict
from glob import glob
import os


#PERFORMING THE CLASSIFICATION USING RANDOM FOREST

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
# from sklearn_xarray.preprocessing import Featurizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
# %matplotlib inline


filelist = sorted(glob(r"/lustre/groups/engstromgrp/max/accra_DIPA/accra_spfeas_10m/*/*.tif"))
fp = r"/lustre/groups/engstromgrp/max/accra_DIPA/TR17.geojson"


fp_rast = filelist
print(fp_rast)

filename_list = []
for file in filelist:
  head, tail = os.path.split(file)
  filenames=tail[:-4]
  filename_list.append(filenames)
print(filename_list)

with gw.open(fp_rast, stack_dim="band", band_names=filename_list) as src:
            df = src.gw.extract(fp,
                       band_names=src.band.values.tolist())


output_csv = "/lustre/groups/engstromgrp/max/accra_DIPA/output/sf_fea.csv"
df.to_csv(output_csv, sep=',', header=True)

x = df[filename_list]
y = df['C_type17']

from sklearn.model_selection import train_test_split
# split the data with 75% in training set
X_train, X_test, y_train, y_test = train_test_split( x, y, 
                                  random_state=0,
                                  train_size=0.75)

# # Use a data pipeline
# pl = Pipeline([('featurizer', Featurizer()),
#                 ('scaler', StandardScaler()),
# #                 ('pca', PCA(n_components = 2)),
#                 ('rf', RandomForestClassifier(n_estimators=1000, max_depth=500, max_features='auto', min_samples_split=3))])


#printing the accuracy metrics
#plugging back the parameters into the model
rfc = RandomForestClassifier(n_estimators=1000, max_depth=500, max_features='auto', min_samples_split=3)

rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, x, y, cv=10)
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())


# with gw.open(fp, stack_dim="band", band_names=filename_list) as src:
#     print(src.crs)
#     X, kmeans = fit(src, df, pl, col='ClassID' )
# print(kmeans)

# fp = filelist
# with gw.open(fp, stack_dim="band", band_names=['blue','green','red', 'nir']) as src:
#     y = fit_predict(src, df, pl, col='ClassID')
#     print(y)
    
# y.sel(band='targ').gw.imshow()