"""
Use K-means clustering latitude & longitude on properties
"""
import pickle
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans

K = range(25, 50, 25)
def km_cluster(K):
    # read properties data
    prop_df = pd.read_csv("input/properties_2016.csv")
    X = prop_df[['latitude', 'longitude']]
    X = X.fillna(X.mean(), inplace=True)
    X = X.astype(np.float32)
    KM = [kmeans(X, k) for k in K]
    pickle.dump(KM, open('output/km_cluster_latlng.pl', 'wb'))

km_cluster(K)
print 'done'
